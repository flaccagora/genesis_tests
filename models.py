import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from transformers import pipeline

class DeformNet_v2(nn.Module):
    def __init__(self, device):
        super(DeformNet_v2, self).__init__()
        self.device = device
        self.set_feature_extractor_v2(device)
    
        
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 3)  # Output 1x3 rotation array of angles
    
    def set_feature_extractor_v2(self, device):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        for param in self.dino.parameters():
            param.requires_grad = False  # Freeze the feature extractor
        self.dinov2.to(device)

    def set_feature_extractor_v3(self, device):

        pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.dino = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map=device,
            )
        for param in self.dino.parameters():
            param.requires_grad = False  # Freeze the feature extractor
        
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.dino(**inputs)
        x = outputs.last_hidden_state  # (batch_size, seq_len, feature_dim)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeformNet_v3(nn.Module):
    def __init__(self, device):
        super(DeformNet_v3, self).__init__()
        self.device = device
        self.set_feature_extractor_v3(device)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 3)  # Output 1x3 rotation array of angles

    def set_feature_extractor_v3(self, device):

        pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.dino = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map=device,
            )
        for param in self.dino.parameters():
            param.requires_grad = False  # Freeze the feature extractor
        
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        outputs = self.dino(**inputs)
        x = outputs.last_hidden_state  # (batch_size, seq_len, feature_dim)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DeformNet_v3_extractor(nn.Module):
    def __init__(self, device):
        super(DeformNet_v3_extractor, self).__init__()
        self.device = device
        self.set_feature_extractor_v3(device)

        self.cnn = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,201,768)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,16,100,384)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,100,384)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,32,50,192)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,50,192)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,64,25,96)

            # Block 4 — further reduce size
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B,128,25,96)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,128,12,48)

            # Block 5 — final reduction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# (B,256,12,48)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,256,6,24)
        )

        # Now flattened size: 256 * 6 * 24 = 36864
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 24, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # final output
        )
    def set_feature_extractor_v3(self, device):
       
        pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.dino = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map=device,
            )

        for param in self.dino.parameters():
            param.requires_grad = False  # Freeze the feature extractor
        
        
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        x = self.dino(**inputs).last_hidden_state.unsqueeze(1)
        # print("EXPECTED 24 201 768 ",x.shape)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    device = "cuda"
    model = DeformNet_v3_extractor(device)
    model.to(device)

    from data import ImageRotationDataset
    import numpy as np

    def get_random_image():
        dataset = ImageRotationDataset("dataset")

        idx = np.random.randint(len(dataset.samples))
        print("index: ", idx)
        image, rotation = dataset.samples[np.random.randint(len(dataset.samples))]
        image = np.load(image)
        rotation = torch.load(rotation)

        return torch.tensor(image, dtype=torch.float16), rotation
    
    image, rotation = get_random_image()
    model(image.to(device))