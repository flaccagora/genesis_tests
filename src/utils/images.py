import matplotlib.pyplot as plt 
import numpy as np

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

