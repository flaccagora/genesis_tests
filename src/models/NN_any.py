"""
Latent Space Regression for 3D Lung Mesh Deformation Prediction
from Endoscopic Images for Robotic Thoracic Surgery.

This module provides two main components:
1. MeshAutoencoder: Learns low-dimensional latent representation of valid lung deformations
2. ImageToMeshPredictor: Predicts mesh deformation from RGB endoscopic images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for mesh vertices.
    
    Processes mesh vertices as a point cloud with permutation invariance.
    Uses shared MLPs followed by max pooling to extract global features.
    
    Args:
        n_vertices (int): Number of vertices in the mesh
        latent_dim (int): Dimension of output latent vector
        hidden_dims (list): Hidden dimensions for shared MLPs
        use_tnet (bool): Whether to use T-Net for input transformation
    """
    
    def __init__(self, n_vertices, latent_dim=32, hidden_dims=None, use_tnet=False):
        super(PointNetEncoder, self).__init__()
        
        self.n_vertices = n_vertices
        self.latent_dim = latent_dim
        self.use_tnet = use_tnet
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        
        # Optional T-Net for input transformation (3x3 transformation matrix)
        if use_tnet:
            self.tnet = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
            )
            self.tnet_fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 9)  # 3x3 transformation matrix
            )
        
        # Shared MLPs (point-wise transformations)
        # Input: [Batch, 3, N_vertices]
        mlp_layers = []
        prev_dim = 3
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, 1),  # 1x1 conv for shared MLP
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim
        
        self.shared_mlp = nn.Sequential(*mlp_layers)
        
        # Final projection to latent space after max pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, mesh):
        """
        Forward pass through PointNet encoder.
        
        Args:
            mesh (torch.Tensor): Input mesh [Batch, N_vertices, 3]
        
        Returns:
            torch.Tensor: Latent vector z [Batch, latent_dim]
        """
        batch_size = mesh.shape[0]
        
        # Transpose for Conv1d: [Batch, N_vertices, 3] -> [Batch, 3, N_vertices]
        x = mesh.transpose(1, 2)  # [Batch, 3, N_vertices]
        
        # Optional T-Net transformation
        if self.use_tnet:
            # Predict transformation matrix
            t_feat = self.tnet(x)  # [Batch, 1024, N_vertices]
            t_feat = torch.max(t_feat, dim=2)[0]  # [Batch, 1024]
            transform = self.tnet_fc(t_feat)  # [Batch, 9]
            transform = transform.view(-1, 3, 3)  # [Batch, 3, 3]
            
            # Add identity matrix for stability
            identity = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
            transform = transform + identity
            
            # Apply transformation: [Batch, 3, 3] @ [Batch, 3, N_vertices]
            x = torch.bmm(transform, x)
        
        # Shared MLPs (point-wise features)
        x = self.shared_mlp(x)  # [Batch, hidden_dims[-1], N_vertices]
        
        # Max pooling to get global feature (permutation invariant)
        global_feature = torch.max(x, dim=2)[0]  # [Batch, hidden_dims[-1]]
        
        # Final projection to latent space
        z = self.fc(global_feature)  # [Batch, latent_dim]
        
        return z


class MeshAutoencoder(nn.Module):
    """
    Mesh Autoencoder - The Deformation Prior
    
    Learns a low-dimensional latent representation of valid lung deformations
    using an encoder-decoder architecture. The decoder acts as a Statistical Shape Model.
    
    Args:
        n_vertices (int): Number of vertices in the mesh (fixed topology)
        latent_dim (int): Dimension of the latent space (default: 32)
        hidden_dims (list): Hidden layer dimensions for encoder/decoder
        encoder_type (str): Type of encoder - 'mlp' or 'pointnet' (default: 'mlp')
        use_tnet (bool): Whether to use T-Net in PointNet encoder (default: False)
    """
    
    def __init__(self, n_vertices, latent_dim=32, hidden_dims=None, encoder_type='mlp', use_tnet=False):
        super(MeshAutoencoder, self).__init__()
        
        self.n_vertices = n_vertices
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        
        # Default hidden layer dimensions if not provided
        if hidden_dims is None:
            if encoder_type == 'pointnet':
                hidden_dims = [64, 128, 256]  # PointNet default
            else:
                hidden_dims = [512, 256, 128]  # MLP default
        
        # ===== ENCODER =====
        if encoder_type == 'pointnet':
            # PointNet-style encoder (permutation invariant)
            self.encoder = PointNetEncoder(
                n_vertices=n_vertices,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                use_tnet=use_tnet
            )
        elif encoder_type == 'mlp':
            # Simple MLP that flattens geometry into latent vector z
            input_dim = n_vertices * 3
            encoder_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            # Final layer to latent space
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'mlp' or 'pointnet'.")
        
        # ===== DECODER =====
        # MLP that expands latent vector back to mesh vertices
        decoder_layers = []
        prev_dim = latent_dim
        input_dim = n_vertices * 3

        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer to reconstruct mesh
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, mesh):
        """
        Encode mesh to latent representation.
        
        Args:
            mesh (torch.Tensor): Input mesh [Batch, N_vertices, 3]
        
        Returns:
            torch.Tensor: Latent vector z [Batch, latent_dim]
        """
        if self.encoder_type == 'pointnet':
            # PointNet encoder expects [Batch, N_vertices, 3]
            z = self.encoder(mesh)
        else:  # mlp
            # MLP encoder expects flattened input [Batch, N_vertices * 3]
            batch_size = mesh.shape[0]
            mesh_flat = mesh.view(batch_size, -1)
            z = self.encoder(mesh_flat)
        return z
    
    def decode(self, z):
        """
        Decode latent vector to mesh.
        
        Args:
            z (torch.Tensor): Latent vector [Batch, latent_dim]
        
        Returns:
            torch.Tensor: Reconstructed mesh [Batch, N_vertices, 3]
        """
        mesh_flat = self.decoder(z)
        # Reshape: [Batch, N_vertices * 3] -> [Batch, N_vertices, 3]
        mesh = mesh_flat.view(-1, self.n_vertices, 3)
        return mesh
    
    def forward(self, mesh):
        """
        Full forward pass: encode then decode.
        
        Args:
            mesh (torch.Tensor): Input mesh [Batch, N_vertices, 3]
        
        Returns:
            tuple: (reconstructed_mesh, latent_z)
                - reconstructed_mesh: [Batch, N_vertices, 3]
                - latent_z: [Batch, latent_dim]
        """
        z = self.encode(mesh)
        reconstructed_mesh = self.decode(z)
        return reconstructed_mesh, z


class ImageToMeshPredictor(nn.Module):
    """
    Image-to-Mesh Predictor - The Inference Engine
    
    Predicts mesh deformation from RGB endoscopic images during surgery.
    Uses a ResNet-18 backbone to extract image features, then maps to the
    latent space learned by the MeshAutoencoder.
    
    Args:
        mesh_decoder (nn.Module): The decoder from a pre-trained MeshAutoencoder
        n_vertices (int): Number of vertices in the mesh (for reshaping output)
        latent_dim (int): Dimension of the latent space (default: 32)
        pretrained_resnet (bool): Whether to use pretrained ResNet-18 (default: True)
    """
    
    def __init__(self, mesh_decoder, n_vertices, latent_dim=32, pretrained_resnet=True):
        super(ImageToMeshPredictor, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_vertices = n_vertices
        
        # ===== BACKBONE: ResNet-18 =====
        resnet18 = models.resnet18(pretrained=pretrained_resnet)
        
        # Remove the final fully connected layer
        # ResNet-18 has 512 features before the final FC layer
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        resnet_output_dim = 512
        
        # ===== HEAD: FC layer to latent space =====
        self.fc_head = nn.Sequential(
            nn.Linear(resnet_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
        
        # ===== DECODER: From MeshAutoencoder (frozen or fine-tunable) =====
        self.mesh_decoder = mesh_decoder
    
    def forward(self, image):
        """
        Forward pass: Image -> ResNet -> Predicted_z -> Decoder -> Predicted_Mesh
        
        Args:
            image (torch.Tensor): RGB image [Batch, 3, 256, 256]
        
        Returns:
            tuple: (predicted_mesh, predicted_z)
                - predicted_mesh: [Batch, N_vertices, 3]
                - predicted_z: [Batch, latent_dim]
        """
        # Extract features using ResNet backbone
        features = self.backbone(image)  # [Batch, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [Batch, 512]
        
        # Map to latent space
        predicted_z = self.fc_head(features)  # [Batch, latent_dim]
        
        # Decode to mesh using pre-trained decoder
        # Decoder outputs flattened [Batch, N_vertices * 3], need to reshape
        mesh_flat = self.mesh_decoder(predicted_z)  # [Batch, N_vertices * 3]
        predicted_mesh = mesh_flat.view(-1, self.n_vertices, 3)  # [Batch, N_vertices, 3]
        
        return predicted_mesh, predicted_z


# ============================================================================
# TRAINING EXAMPLE - Stage 1: MeshAutoencoder Training
# ============================================================================

def train_stage1_example(encoder_type='mlp', use_tnet=False):
    """
    Training loop for Stage 1: MeshAutoencoder
    
    Trains the mesh autoencoder to learn a low-dimensional latent representation
    of valid lung deformations. The trained decoder will be used in Stage 2.
    
    Args:
        encoder_type (str): 'mlp' or 'pointnet'
        use_tnet (bool): Whether to use T-Net in PointNet (only if encoder_type='pointnet')
    
    Returns:
        MeshAutoencoder: Trained autoencoder model
    """
    
    # ===== HYPERPARAMETERS =====
    N_VERTICES = 1000  # Example: lung mesh with 1000 vertices
    LATENT_DIM = 32
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(f"STAGE 1: Training MeshAutoencoder ({encoder_type.upper()})")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Encoder Type: {encoder_type}")
    if encoder_type == 'pointnet':
        print(f"T-Net: {use_tnet}")
    print(f"Latent Dimension: {LATENT_DIM}")
    print(f"Number of Vertices: {N_VERTICES}")
    print()
    
    # ===== STEP 1: Initialize MeshAutoencoder =====
    print("[Stage 1] Initializing MeshAutoencoder...")
    model = MeshAutoencoder(
        n_vertices=N_VERTICES,
        latent_dim=LATENT_DIM,
        encoder_type=encoder_type,
        use_tnet=use_tnet
    ).to(DEVICE)
    
    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")
    
    # ===== STEP 2: Setup Training =====
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # ===== STEP 3: Training Loop =====
    print("\n[Training] Starting Stage 1 training...\n")
    
    best_loss = float('inf')
    num_batches_train = 50  # Dummy: assume 50 batches per epoch
    num_batches_val = 10    # Dummy: assume 10 validation batches
    
    for epoch in range(NUM_EPOCHS):
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        
        for batch_idx in range(num_batches_train):
            # ===== DUMMY DATA =====
            # In practice, replace with actual DataLoader
            # Simulating deformed lung meshes
            input_meshes = torch.randn(BATCH_SIZE, N_VERTICES, 3).to(DEVICE)
            
            # ===== FORWARD PASS =====
            reconstructed_meshes, latent_z = model(input_meshes)
            
            # ===== COMPUTE LOSS =====
            # MSE between input and reconstructed meshes
            loss = criterion(reconstructed_meshes, input_meshes)
            
            # ===== BACKWARD PASS =====
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / num_batches_train
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx in range(num_batches_val):
                # Dummy validation data
                input_meshes = torch.randn(BATCH_SIZE, N_VERTICES, 3).to(DEVICE)
                
                reconstructed_meshes, latent_z = model(input_meshes)
                loss = criterion(reconstructed_meshes, input_meshes)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_batches_val
        
        # ===== EPOCH SUMMARY =====
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # ===== SAVE BEST MODEL =====
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint_path = f'mesh_autoencoder_{encoder_type}_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'encoder_type': encoder_type,
                'n_vertices': N_VERTICES,
                'latent_dim': LATENT_DIM,
            }, checkpoint_path)
            print(f"  → Best model saved: {checkpoint_path} (Val Loss: {avg_val_loss:.6f})")
        
        # ===== SAVE CHECKPOINT PERIODICALLY =====
        if (epoch + 1) % 25 == 0:
            checkpoint_path = f'mesh_autoencoder_{encoder_type}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'encoder_type': encoder_type,
                'n_vertices': N_VERTICES,
                'latent_dim': LATENT_DIM,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    print("\n[Training Complete] Stage 1 training finished!")
    print(f"Best validation loss: {best_loss:.6f}")
    
    # ===== TEST LEARNED REPRESENTATION =====
    print("\n[Testing] Evaluating learned latent representation...")
    model.eval()
    with torch.no_grad():
        # Test reconstruction
        test_mesh = torch.randn(1, N_VERTICES, 3).to(DEVICE)
        recon_mesh, z = model(test_mesh)
        recon_error = torch.mean((recon_mesh - test_mesh) ** 2).item()
        
        print(f"  Test mesh shape: {test_mesh.shape}")
        print(f"  Latent z shape: {z.shape}")
        print(f"  Latent z mean: {z.mean().item():.4f}, std: {z.std().item():.4f}")
        print(f"  Reconstruction error (MSE): {recon_error:.6f}")
        
        # Test latent space interpolation
        print("\n  Testing latent space interpolation...")
        mesh1 = torch.randn(1, N_VERTICES, 3).to(DEVICE)
        mesh2 = torch.randn(1, N_VERTICES, 3).to(DEVICE)
        z1 = model.encode(mesh1)
        z2 = model.encode(mesh2)
        
        # Interpolate in latent space
        alpha = 0.5
        z_interp = alpha * z1 + (1 - alpha) * z2
        mesh_interp = model.decode(z_interp)
        print(f"  Interpolated mesh shape: {mesh_interp.shape}")
    
    return model


# ============================================================================
# TRAINING EXAMPLE - Stage 2: Image-to-Mesh Training
# ============================================================================

def train_stage2_example():
    """
    Dummy training loop for Stage 2: Image-to-Mesh Prediction
    
    Assumes:
    - MeshAutoencoder is already trained and decoder is frozen
    - Training data consists of (image, ground_truth_mesh) pairs
    """
    
    # ===== HYPERPARAMETERS =====
    N_VERTICES = 1000  # Example: lung mesh with 1000 vertices
    LATENT_DIM = 32
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {DEVICE}")
    
    # ===== STEP 1: Load Pre-trained MeshAutoencoder =====
    print("\n[Stage 1] Loading pre-trained MeshAutoencoder...")
    mesh_autoencoder = MeshAutoencoder(
        n_vertices=N_VERTICES,
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    # In practice, you would load pre-trained weights here:
    # mesh_autoencoder.load_state_dict(torch.load('mesh_autoencoder.pth'))
    
    # Freeze the decoder (it's our learned deformation prior)
    for param in mesh_autoencoder.decoder.parameters():
        param.requires_grad = False
    
    mesh_autoencoder.eval()  # Set to eval mode since decoder is frozen
    
    # ===== STEP 2: Initialize Image-to-Mesh Predictor =====
    print("[Stage 2] Initializing Image-to-Mesh Predictor...")
    model = ImageToMeshPredictor(
        mesh_decoder=mesh_autoencoder.decoder,
        n_vertices=N_VERTICES,
        latent_dim=LATENT_DIM,
        pretrained_resnet=True
    ).to(DEVICE)
    
    # ===== STEP 3: Setup Training =====
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ===== STEP 4: Dummy Training Loop =====
    print("\n[Training] Starting Stage 2 training...\n")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 10  # Dummy: assume 10 batches per epoch
        
        for batch_idx in range(num_batches):
            # ===== DUMMY DATA =====
            # In practice, replace with actual DataLoader
            # Simulating: (RGB images, ground-truth deformed meshes)
            dummy_images = torch.randn(BATCH_SIZE, 3, 256, 256).to(DEVICE)
            dummy_gt_meshes = torch.randn(BATCH_SIZE, N_VERTICES, 3).to(DEVICE)
            
            # ===== FORWARD PASS =====
            predicted_mesh, predicted_z = model(dummy_images)
            
            # ===== COMPUTE LOSS =====
            # MSE between predicted mesh vertices and ground-truth mesh vertices
            loss = criterion(predicted_mesh.view(BATCH_SIZE, -1), dummy_gt_meshes.view(BATCH_SIZE, -1))
            
            # ===== BACKWARD PASS =====
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # ===== EPOCH SUMMARY =====
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.6f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # ===== VALIDATION (Optional) =====
        # In practice, evaluate on validation set here
        
        # ===== SAVE CHECKPOINT =====
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'image_to_mesh_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    print("\n[Training Complete] Stage 2 training finished!")
    
    # ===== INFERENCE EXAMPLE =====
    print("\n[Inference] Testing prediction on a single image...")
    model.eval()
    with torch.no_grad():
        test_image = torch.randn(1, 3, 256, 256).to(DEVICE)
        predicted_mesh, predicted_z = model(test_image)
        print(f"  Input image shape: {test_image.shape}")
        print(f"  Predicted latent z shape: {predicted_z.shape}")
        print(f"  Predicted mesh shape: {predicted_mesh.shape}")
    
    return model


if __name__ == "__main__":
    """
    Complete demonstration of the two-stage training pipeline.
    
    Usage:
        python NN_any.py                    # Run full demo
        python NN_any.py --stage1           # Only Stage 1 (MeshAutoencoder)
        python NN_any.py --stage2           # Only Stage 2 (Image-to-Mesh)
        python NN_any.py --encoder pointnet # Use PointNet encoder
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D Mesh Deformation Prediction Model')
    parser.add_argument('--stage1', action='store_true', help='Run Stage 1 training only')
    parser.add_argument('--stage2', action='store_true', help='Run Stage 2 training only')
    parser.add_argument('--encoder', type=str, default='mlp', choices=['mlp', 'pointnet'], 
                        help='Encoder type for Stage 1 (default: mlp)')
    parser.add_argument('--tnet', action='store_true', help='Use T-Net in PointNet encoder')
    parser.add_argument('--skip-demo', action='store_true', help='Skip architecture comparison demo')
    args = parser.parse_args()
    
    # Default: run everything if no specific stage is selected
    run_stage1 = args.stage1 or not (args.stage1 or args.stage2)
    run_stage2 = args.stage2 or not (args.stage1 or args.stage2)
    
    # ===== DEMO: Compare MLP vs PointNet Encoders =====
    if not args.skip_demo:
        print("=" * 60)
        print("DEMO: Comparing MLP and PointNet Encoders")
        print("=" * 60)
        
        N_VERTICES = 1000
        LATENT_DIM = 32
        BATCH_SIZE = 4
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test data
        test_mesh = torch.randn(BATCH_SIZE, N_VERTICES, 3).to(DEVICE)
        
        # Test MLP Encoder
        print("\n1. Testing MLP Encoder:")
        mlp_autoencoder = MeshAutoencoder(
            n_vertices=N_VERTICES,
            latent_dim=LATENT_DIM,
            encoder_type='mlp'
        ).to(DEVICE)
        
        with torch.no_grad():
            recon_mesh_mlp, z_mlp = mlp_autoencoder(test_mesh)
        
        print(f"   Input mesh shape: {test_mesh.shape}")
        print(f"   Latent z shape: {z_mlp.shape}")
        print(f"   Reconstructed mesh shape: {recon_mesh_mlp.shape}")
        print(f"   MLP Autoencoder params: {sum(p.numel() for p in mlp_autoencoder.parameters()):,}")
        
        # Test PointNet Encoder
        print("\n2. Testing PointNet Encoder:")
        pointnet_autoencoder = MeshAutoencoder(
            n_vertices=N_VERTICES,
            latent_dim=LATENT_DIM,
            encoder_type='pointnet',
            use_tnet=False
        ).to(DEVICE)
        
        with torch.no_grad():
            recon_mesh_pn, z_pn = pointnet_autoencoder(test_mesh)
        
        print(f"   Input mesh shape: {test_mesh.shape}")
        print(f"   Latent z shape: {z_pn.shape}")
        print(f"   Reconstructed mesh shape: {recon_mesh_pn.shape}")
        print(f"   PointNet Autoencoder params: {sum(p.numel() for p in pointnet_autoencoder.parameters()):,}")
        
        # Test PointNet Encoder with T-Net
        print("\n3. Testing PointNet Encoder with T-Net:")
        pointnet_tnet_autoencoder = MeshAutoencoder(
            n_vertices=N_VERTICES,
            latent_dim=LATENT_DIM,
            encoder_type='pointnet',
            use_tnet=True
        ).to(DEVICE)
        
        with torch.no_grad():
            recon_mesh_pn_t, z_pn_t = pointnet_tnet_autoencoder(test_mesh)
        
        print(f"   Input mesh shape: {test_mesh.shape}")
        print(f"   Latent z shape: {z_pn_t.shape}")
        print(f"   Reconstructed mesh shape: {recon_mesh_pn_t.shape}")
        print(f"   PointNet+T-Net Autoencoder params: {sum(p.numel() for p in pointnet_tnet_autoencoder.parameters()):,}")
        
        print("\n")
    
    # ===== STAGE 1: Train MeshAutoencoder =====
    if run_stage1:
        print("\n" + "=" * 60)
        print("STARTING STAGE 1: MeshAutoencoder Training")
        print("=" * 60 + "\n")
        
        trained_autoencoder = train_stage1_example(
            encoder_type=args.encoder,
            use_tnet=args.tnet
        )
        
        print("\n" + "=" * 60)
        print("STAGE 1 COMPLETE")
        print("=" * 60 + "\n")
    
    # ===== STAGE 2: Train Image-to-Mesh Predictor =====
    if run_stage2:
        print("\n" + "=" * 60)
        print("STARTING STAGE 2: Image-to-Mesh Training")
        print("=" * 60 + "\n")
        
        trained_predictor = train_stage2_example()
        
        print("\n" + "=" * 60)
        print("STAGE 2 COMPLETE")
        print("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace dummy data with your actual lung mesh dataset")
    print("2. Train Stage 1 to learn deformation prior")
    print("3. Freeze decoder and train Stage 2 with (image, mesh) pairs")
    print("4. Deploy for real-time surgical guidance")
    print("=" * 60)