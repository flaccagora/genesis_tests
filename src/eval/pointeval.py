import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, List

from train.train_encoder.module import MeshAutoencoderLightningModule
from utils.configurator import apply_overrides

def load_autoencoder_model(
    checkpoint_path: str,
    n_vertices: int,
    latent_dim: int = 32,
    encoder_type: str = "mlp",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> MeshAutoencoderLightningModule:
    """Load the MeshAutoencoder from a Lightning checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    model = MeshAutoencoderLightningModule.load_from_checkpoint(
        checkpoint_path,
        n_vertices=n_vertices,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        map_location=device
    )
    model.to(device)
    model.eval()
    return model

def visualize_reconstruction(original_points: np.ndarray, reconstructed_points: np.ndarray, title: str = "Reconstruction Visualization"):
    """Visualize original and reconstructed points side-by-side using matplotlib."""
    fig = plt.figure(figsize=(15, 7))
    
    # Original Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c='blue', s=1, alpha=0.5)
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Reconstructed Point Cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], c='red', s=1, alpha=0.5)
    ax2.set_title("Reconstructed Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Set equal aspect ratio for better comparison
    for ax in [ax1, ax2]:
        # Simple method to get somewhat equal axes
        X, Y, Z = original_points[:, 0], original_points[:, 1], original_points[:, 2]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def run_point_evaluation(
    model: MeshAutoencoderLightningModule,
    data_dir: str,
    device: torch.device,
    num_samples: int = 5
):
    """Run point cloud reconstruction evaluation on samples from data_dir."""
    npy_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.npy')]
    if not npy_files:
        print(f"No .npy files found in {data_dir}")
        return

    print(f"Found {len(npy_files)} files in {data_dir}. Evaluating {num_samples} samples.")
    
    indices = np.random.choice(len(npy_files), min(num_samples, len(npy_files)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            file_path = os.path.join(data_dir, npy_files[idx])
            print(f"[{i+1}/{num_samples}] Evaluating {file_path}...")
            
            # Load original points
            original_points = np.load(file_path)
            # Shapes might need adjustment [N, 3] -> [1, N, 3]
            original_tensor = torch.from_numpy(original_points).float().to(device).unsqueeze(0)
            
            # Forward pass
            reconstructed_tensor, _ = model(original_tensor)
            reconstructed_points = reconstructed_tensor.squeeze(0).cpu().numpy()
            
            # Compute Reconstruction RMSE
            rmse = np.sqrt(np.mean((original_points - reconstructed_points)**2))
            print(f"  Reconstruction RMSE: {rmse:.6f}")
            
            # Visualize
            visualize_reconstruction(
                original_points, 
                reconstructed_points, 
                title=f"Sample: {npy_files[idx]} | RMSE: {rmse:.6f}"
            )

if __name__ == "__main__":
    # Settings (can be overridden via system args)
    checkpoint_path = "lightning_logs/train_encoder/mesh_autoencoder/wo54dh4a/checkpoints/last.ckpt"  # MUST BE PROVIDED
    data_dir = "datasets/lungs_bronchi/particles"
    n_vertices = 4461
    latent_dim = 32
    encoder_type = "pointnet"
    num_samples = 5
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration override support
    config_keys = [k for k, v in globals().items() 
                   if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    apply_overrides(globals())
    
    if checkpoint_path is None:
        print("Error: checkpoint_path must be provided via command line as 'checkpoint_path=path/to/ckpt'")
        print("Example: PYTHONPATH=src python -m src.eval.pointeval checkpoint_path=lightning_logs/train_encoder/checkpoints/last.ckpt")
        exit(1)
        
    device = torch.device(device_name)
    
    try:
        # Load model
        model = load_autoencoder_model(
            checkpoint_path=checkpoint_path,
            n_vertices=n_vertices,
            latent_dim=latent_dim,
            encoder_type=encoder_type,
            device=device
        )
        
        # Run evaluation
        run_point_evaluation(
            model=model,
            data_dir=data_dir,
            device=device,
            num_samples=num_samples
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
