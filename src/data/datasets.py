import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np

class ImageRotationDataset(Dataset):
    def __init__(self, root_dir, rgb, depth, transform=None):
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
        depth = to_tensor(depth)

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


def show_image(image_array):
    """
    Display an image from a numpy array.
    
    Parameters:
    image_array (np.ndarray): Image array with shape (height, width, channels)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.title('Image Display')
    plt.show()
    
def show_images(*images):
    """
    Display up to 6 images in a 2x3 grid.

    Parameters:
    *images: any number of numpy image arrays (1 to 6).
    """

    num_images = len(images)
    assert 1 <= num_images <= 6, "You must pass between 1 and 6 images."

    cols = 2
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(15, 8))

    for i, img in enumerate(images, start=1):

        # Choose subplot position
        plt.subplot(rows, cols, i)

        # Handle grayscale, H×W×1, and RGB
        if img.ndim == 2:  # grayscale
            plt.imshow(img, cmap='gray')
        elif img.ndim == 3 and img.shape[2] == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)

        plt.axis('off')
        plt.title(f"Image {i}")

    plt.tight_layout()
    plt.show()

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

