import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


def rotation_6d_to_matrix(rot_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    # Gram-Schmidt orthogonalization
    x = rot_6d[:3]
    y = rot_6d[3:]
    
    x = F.normalize(x, dim=-1)
    y = y - (x * y).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1)
    z = torch.cross(x, y, dim=-1)
    
    return torch.stack([x, y, z], dim=-1)


def geodesic_loss(rot_pred_6d, rot_target):
    """
    Geodesic loss on SO(3).
    Args:
        rot_pred_6d: predicted 6D rotation (B, 6)
        rot_target: target 3x3 rotation matrix (B, 3, 3)
    """
    batch_size = rot_pred_6d.shape[0]
    
    # Convert predictions to rotation matrices
    rot_pred_6d_expanded = rot_pred_6d.reshape(batch_size, 6)
    rot_pred = torch.stack([rotation_6d_to_matrix(rot_pred_6d_expanded[i]) for i in range(batch_size)])
    
    # Frobenius norm loss on rotation matrices
    loss = torch.norm(rot_pred - rot_target, dim=[1, 2]).mean()
    
    return loss


class RotationPredictor(nn.Module):
    """Rotation prediction model using DinoV3 backbone."""
    
    def __init__(self, backbone_dim=768, hidden_dim=512):
        """
        Args:
            backbone_dim: output dim of DinoV3 backbone (usually 768)
            hidden_dim: hidden dimension of MLP head
        """
        super().__init__()
        
        # Try to load DinoV3, fallback to ResNet50 if not available
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 768
        except:
            print("DinoV3 not available, using ResNet50 instead")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 2048
        
        # Prediction head: MLP
        self.head = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6)  # 6D rotation representation
        )
    
    def forward(self, x):
        """
        Args:
            x: input images (B, 3, H, W)
        Returns:
            6D rotation representation (B, 6)
        """
        with torch.no_grad():
            features = self.backbone(x)
        
        # Global average pooling if needed
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])
        
        rot_6d = self.head(features)
        
        # Normalize to ensure valid rotation representation
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        
        return rot_6d


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, rot_6d) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        rot_6d = rot_6d.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        rot_pred_6d = model(images)
        
        # Convert 6D to matrix for loss computation
        batch_size = rot_pred_6d.shape[0]
        rot_target = torch.stack([rotation_6d_to_matrix(rot_6d[i:i+1].reshape(6)).squeeze(0) 
                                   for i in range(batch_size)])
        
        # Compute loss
        loss = geodesic_loss(rot_pred_6d, rot_target.to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, rot_6d in tqdm(val_loader):
            images = images.to(device)
            rot_6d = rot_6d.to(device)
            
            rot_pred_6d = model(images)
            
            batch_size = rot_pred_6d.shape[0]
            rot_target = torch.stack([rotation_6d_to_matrix(rot_6d[i:i+1].reshape(6)).squeeze(0) 
                                       for i in range(batch_size)])
            
            loss = geodesic_loss(rot_pred_6d, rot_target.to(device))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    hidden_dim = 512
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # TODO: Replace with your actual dataset
    from data import ImageRotationDataset
    # Create datasets and dataloaders
    train_dataset = ImageRotationDataset("datasets/data_Torus_5",transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = RotationPredictor(hidden_dim=hidden_dim).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for (idx, (image, rotation)) in enumerate(train_loader):
        image = image.to(device)
        rotation = rotation.to(device)
        print("Image shape:", image.shape)
        print("Rotation shape:", rotation.shape)
        output = model(image)
        print(output)
        # rot_target = torch.stack([rotation_6d_to_matrix(rot_6d[i:i+1].reshape(6)).squeeze(0) 
        #                            for i in range(batch_size)])
        
        # Compute loss
        loss = geodesic_loss(output, rotation.to(device))
        print(loss)
        break

    # # Training loop
    # # best_val_loss = float('inf')
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, device)
    #     # val_loss = validate(model, val_loader, device)
    #     scheduler.step()
        
    #     print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}") #, Val Loss: {val_loss:.6f}")
        
    #     # if val_loss < best_val_loss:
    #     #     best_val_loss = val_loss
    #     #     torch.save(model.state_dict(), 'best_rotation_model.pt')
    #     #     print(f"  → Saved best model")
    
    # print("Training complete!")


if __name__ == "__main__":
    main()