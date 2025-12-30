import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np

class ImageRotationDataset(Dataset):
    def __init__(self, root_dir, rgb=True, depth=False, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder containing images and .npy rotation matrices.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.root_dir = root_dir
        self.rgb = rgb
        self.depth = depth
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.root_dir):
            if fname.lower().startswith(("rgb")):
                base = os.path.splitext(fname)[0]
                rgb_path = os.path.join(self.root_dir, fname)
                depth_path = os.path.join(self.root_dir, f"{base.replace('rgb', 'depth')}.npy")
                rot_path = os.path.join(self.root_dir, f"{base.replace('rgb', 'rotation')}.th")
                if os.path.exists(rot_path):
                    samples.append((rgb_path, depth_path, rot_path))
    
        if not samples:
            raise FileNotFoundError(f"No RGB files found in directory {self.root_dir}")
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, rot_path = self.samples[idx]

        # --- Load arrays ---
        rgb = np.load(rgb_path)
        depth = np.load(depth_path)

        # --- Convert to tensor & apply transforms ---
        to_tensor = transforms.ToTensor()

        rgb = self.transform(rgb) if self.transform else to_tensor(rgb)
        reshape = transforms.Resize((224, 224))
        depth = reshape(to_tensor(depth))

        # --- Build RGB-D tensor if requested ---
        if self.rgb and self.depth:
            # Ensure depth has channel dimension (1, H, W)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)

            # Concatenate to RGBD (4, H, W)
            rgbd = torch.cat([rgb, depth], dim=0)
        else:
            # Fall back to using whichever data is requested
            rgbd = rgb if self.rgb else depth

        # --- Load target rotation matrix ---
        rotation_matrix = torch.load(rot_path)

        return rgbd, rotation_matrix

class ANYDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder containing subdirectories:
                - RGB/: RGB images as .npy files
                - depth/: Depth maps as .npy files
                - particles/: Particle positions as .npy files
                - rotation/: Rotation matrices as .npy files
                - actu/: Actuator data as .npy files
            transform (callable, optional): Optional transform to be applied to RGB images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Discover all available samples by scanning the RGB directory."""
        rgb_dir = os.path.join(self.root_dir, "RGB")
        
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        
        samples = []
        for fname in sorted(os.listdir(rgb_dir)):
            if fname.endswith('.npy'):
                # Extract the index (e.g., "0.npy" -> "0")
                idx = os.path.splitext(fname)[0]
                
                # Build paths for all data components
                rgb_path = os.path.join(self.root_dir, "RGB", f"{idx}.npy")
                depth_path = os.path.join(self.root_dir, "depth", f"{idx}.npy")
                particles_path = os.path.join(self.root_dir, "particles", f"{idx}.npy")
                rotation_path = os.path.join(self.root_dir, "rotation", f"{idx}.npy")
                actu_path = os.path.join(self.root_dir, "actu", f"{idx}.npy")
                
                # Verify all files exist
                if all(os.path.exists(p) for p in [rgb_path, depth_path, particles_path, rotation_path, actu_path]):
                    samples.append({
                        'rgb': rgb_path,
                        'depth': depth_path,
                        'particles': particles_path,
                        'rotation': rotation_path,
                        'actu': actu_path
                    })
        
        if not samples:
            raise FileNotFoundError(f"No complete samples found in directory {self.root_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: Dictionary containing:
                - 'rgb': Tensor of shape (C, H, W) or (H, W, C) depending on transform
                - 'depth': Tensor of shape (H, W)
                - 'particles': Tensor of particle positions
                - 'rotation': Tensor of rotation matrix
                - 'actu': Tensor of actuator data
        """
        sample_paths = self.samples[idx]
        
        # Load all numpy arrays
        rgb = np.load(sample_paths['rgb'])
        depth = np.load(sample_paths['depth'])
        particles = np.load(sample_paths['particles'])
        rotation = np.load(sample_paths['rotation'])
        actu = np.load(sample_paths['actu'])
        
        # Apply transform to RGB if provided
        if self.transform:
            rgb = self.transform(rgb)
        else:
            # Convert to tensor with default ToTensor transform
            rgb = torch.from_numpy(rgb)
            # If RGB is (H, W, C), permute to (C, H, W)
            if rgb.dim() == 3 and rgb.shape[-1] in [1, 3, 4]:
                rgb = rgb.permute(2, 0, 1)
        
        # Convert other arrays to tensors
        depth = torch.from_numpy(depth)
        particles = torch.from_numpy(particles)
        rotation = torch.from_numpy(rotation)
        actu = torch.from_numpy(actu).float()
        
        return {
            'rgb': rgb,
            'depth': depth,
            'particles': particles,
            'rotation': rotation,
            'actu': actu
        }

class ImageActuationRotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder containing subdirectories:
                - RGB/: RGB images as .npy files
                - actu/: Actuator data as .npy files
                - rotation/: Rotation matrices as .npy files
                - particles/: Particle positions as .npy files
            transform (callable, optional): Optional transform to be applied to RGB images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        rgb_dir = os.path.join(self.root_dir, "RGB")
        actu_dir = os.path.join(self.root_dir, "actu")
        rot_dir = os.path.join(self.root_dir, "rotation")
        particles_dir = os.path.join(self.root_dir, "particles")
        
        if not os.path.exists(rgb_dir) or not os.path.exists(actu_dir) or not os.path.exists(rot_dir) or not os.path.exists(particles_dir):
             raise FileNotFoundError(f"RGB, actu, rotation or particles directory not found in {self.root_dir}")

        samples = []
        for fname in sorted(os.listdir(rgb_dir)):
            if fname.endswith('.npy'):
                idx = os.path.splitext(fname)[0]
                rgb_path = os.path.join(rgb_dir, f"{idx}.npy")
                actu_path = os.path.join(actu_dir, f"{idx}.npy")
                rot_path = os.path.join(rot_dir, f"{idx}.npy")
                particles_path = os.path.join(particles_dir, f"{idx}.npy")
                
                if os.path.exists(actu_path) and os.path.exists(rot_path) and os.path.exists(particles_path):
                    samples.append((rgb_path, actu_path, rot_path, particles_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, actu_path, rot_path, particles_path = self.samples[idx]
        rgb = np.load(rgb_path)
        actu = np.load(actu_path)
        rot = np.load(rot_path)
        particles = np.load(particles_path)

        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).float()
            if rgb.dim() == 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)

        actu = torch.from_numpy(actu).float()
        rot = torch.from_numpy(rot).float()
        particles = torch.from_numpy(particles).float()
        return rgb, actu, rot, particles
        
def create_dataloader(
    root_dir, batch_size=32, shuffle=True, num_workers=0, img_size=224):
    """
    Create a PyTorch DataLoader for (image, rotation_matrix) pairs.
    """
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(),
    ])

    dataset = ImageRotationDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    from utils.images import show_image
    import argparse
    parser = argparse.ArgumentParser(description="Test ANYDataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--any", action="store_true", default=False,
                        help="Test ANYDataset")
    args = parser.parse_args()
    
    # Test ANYDataset
    print("=" * 50)
    print("Testing ANYDataset")
    print("=" * 50)
    
    dataset_path = args.dataset_path
    
    if args.any:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Must convert numpy to tensor FIRST
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # Load the dataset
            dataset = ANYDataset(dataset_path, transform=transform)
            print(f"\n✓ Dataset loaded successfully!")
            print(f"  Total samples: {len(dataset)}")
            
            # Test a single sample
            print(f"\n--- Testing sample 0 ---")
            sample = dataset[0]
            
            print(f"\nData shapes:")
            print(f"  RGB:       {sample['rgb'].shape}")
            print(f"  Depth:     {sample['depth'].shape}")
            print(f"  Particles: {sample['particles'].shape}")
            print(f"  Rotation:  {sample['rotation'].shape}")
            print(f"  Actuator:  {sample['actu'].shape}")
            
            print(f"\nData types:")
            print(f"  RGB:       {sample['rgb'].dtype}")
            print(f"  Depth:     {sample['depth'].dtype}")
            print(f"  Particles: {sample['particles'].dtype}")
            print(f"  Rotation:  {sample['rotation'].dtype}")
            print(f"  Actuator:  {sample['actu'].dtype}")
            
            # Show the RGB image
            print(f"\n--- Displaying RGB image ---")
            show_image(sample['rgb'])

            print(f"\n--- Displaying Depth image ---")
            show_image(sample['depth'])
            
            # Test with DataLoader
            print(f"\n--- Testing DataLoader ---")
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            
            for batch_idx, batch in enumerate(dataloader):
                print(f"\nBatch {batch_idx}:")
                print(f"  RGB batch shape:       {batch['rgb'].shape}")
                print(f"  Depth batch shape:     {batch['depth'].shape}")
                print(f"  Particles batch shape: {batch['particles'].shape}")
                print(f"  Rotation batch shape:  {batch['rotation'].shape}")
                print(f"  Actuator batch shape:  {batch['actu'].shape}")
                
                if batch_idx == 0:  # Just show first batch
                    break
            
            print(f"\n✓ All tests passed!")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        dataset = ImageRotationDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  RGB batch shape:       {batch['rgb'].shape}")
            print(f"  Depth batch shape:     {batch['depth'].shape}")
            print(f"  Particles batch shape: {batch['particles'].shape}")
            print(f"  Rotation batch shape:  {batch['rotation'].shape}")
            print(f"  Actuator batch shape:  {batch['actu'].shape}")
            
            if batch_idx == 0:  # Just show first batch
                break
        
        print(f"\n✓ All tests passed!")