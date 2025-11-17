from models import DeformNet_v2
from models import DeformNet_v3_extractor as DeformNet_v3
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class ImageRotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder containing images and .npy rotation matrices.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.root_dir):
            if fname.lower().endswith((".npy", ".jpg", ".jpeg")):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.root_dir, fname)
                rot_path = os.path.join(self.root_dir, f"{base.replace("image", "rotation")}.th")
                if os.path.exists(rot_path):
                    samples.append((img_path, rot_path))
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, rot_path = self.samples[idx]
        # Load image
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Load rotation matrix (1x3)
        rotation_matrix = torch.load(rot_path)

        return torch.tensor(image,dtype=torch.float16), rotation_matrix

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

def train(epochs, bs, dino="v3", pretrained_model=None, compile=False):
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE ", device)
    dataloader = create_dataloader('dataset', batch_size=bs)
    
    if dino == "v3":
        model = DeformNet_v3
    elif dino == "v2":
        model = DeformNet_v2
    else:
        raise ValueError


    if pretrained_model==None:
        deformnet = model(device=device)
    else:
        deformnet = pretrained_model
    deformnet.to(device)
    if compile:
        torch.compile(deformnet)

    num_total = sum(p.numel() for p in deformnet.parameters())
    num_trainable = sum(p.numel() for p in deformnet.parameters() if p.requires_grad)

    print(f"Total parameters: {num_total}")
    print(f"Trainable parameters: {num_trainable}")


    optimizer = torch.optim.Adam(deformnet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, rotation_matrices in epoch_bar:
            optimizer.zero_grad()
            outputs = deformnet(images.to(device))
            loss = criterion(outputs, rotation_matrices.to(device))
            loss.backward()
            optimizer.step()

            # update tqdm postfix
            epoch_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        if (epoch+1) % 5 == 0:
            torch.save(deformnet.state_dict(), f"trained_{dino}_{epoch+1}_8k.pth")

    return deformnet

if __name__ == "__main__":

    # -----------------------------------------------------------------------------
    # I/O
    out_dir = 'out'
    init_from = 'scratch' # 'scratch' or 'not init_from == "scratch"' or 'gpt2*'
    # train
    epochs = 10
    batch_size = 128
    # model
    dino="v3"
    # data
    dataset = 'openwebtext'
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    if dino == "v3":
        model = DeformNet_v3
    elif dino == "v2":
        model = DeformNet_v2
    else:
        raise ValueError

    trained_model = None
    if not init_from == "scratch":
        trained_model = model(device)
        trained_model.load_state_dict(torch.load(f"trained_{dino}_{epochs}_8k.pth"))


    trained_model = train(epochs=epochs, bs=batch_size, pretrained_model=trained_model, compile=False)
    torch.save(trained_model.state_dict(), f"trained_{dino}_{epochs}_8k.pth")
   



