import matplotlib.pyplot as plt 
import numpy as np
import torch

def show_image(image_array, is_depth=False):
    """
    Display an image or depth map from a numpy array or torch tensor.
    
    Parameters:
    image_array (np.ndarray or torch.Tensor): Image array or tensor.
    is_depth (bool): Whether to treat the input as a depth map.
    """
    print("SHAPE before: ", image_array.shape)

    if isinstance(image_array, torch.Tensor):
        if image_array.ndim == 3 and image_array.shape[0] in [1, 3]:
            image_array = image_array.permute(1, 2, 0)
        image_array = image_array.cpu().numpy()
    print("SHAPE after: ", image_array.shape)
    plt.figure(figsize=(10, 8))
    if is_depth:
        plt.imshow(image_array, cmap='gray')
    else:
        plt.imshow(image_array)
    plt.axis('off')
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

