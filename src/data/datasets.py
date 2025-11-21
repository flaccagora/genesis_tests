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
    
    def get_random_image(dataset, depth = False):
        dataset = ImageRotationDataset(dataset, depth=depth)
        idx = np.random.randint(len(dataset))
        print(f"Selected index: {idx}")
        rgbd, rotation = dataset[idx]

        return rgbd, rotation

    # Example usage
    dataset_path = "datasets/data_lungs_5"
    while True:
        img , rot = get_random_image(dataset_path, depth=False)
        show_image(img)

    
    
    
    # dataloader = create_dataloader(dataset_path, batch_size=16, img_size=128)

    # for batch_idx, (images, rotations) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Images shape: {images.shape}")
    #     print(f"  Rotations shape: {rotations.shape}")
    #     show_image(images[0])  # Show first image in the batch
    #     if batch_idx == 1:  # Just show first two batches
    #         break

    