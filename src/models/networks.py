import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

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
        self.dino.to(device)

    def _resolve_device(self):
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device(self.device)

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
        runtime_device = self._resolve_device()
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(runtime_device)
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
        self.fc2 = nn.Linear(521,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128, 3)  # Output 1x3 rotation array of angles

    def _resolve_device(self):
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device(self.device)

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
        runtime_device = self._resolve_device()
        inputs = self.processor(images=x, return_tensors="pt", 
                                do_rescale=False,
                                do_resize=True,).to(runtime_device)

        outputs = self.dino(**inputs)
        # outputs = self.dino(x)
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
    def _resolve_device(self):
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device(self.device)

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
        runtime_device = self._resolve_device()
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(runtime_device)
        x = self.dino(**inputs).last_hidden_state.unsqueeze(1)
        # print("EXPECTED 24 201 768 ",x.shape)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x
    
class RGBDNN(nn.Module):
    def __init__(self, device):
        super(RGBDNN, self).__init__()
        self.device = device
        self.image_size = (480, 640)
        self.patch_size = 16
        self.patch_h = self.image_size[0] // self.patch_size
        self.patch_w = self.image_size[1] // self.patch_size
        self.set_feature_extractor_v3(device)
        self.rgb_feature_dim = self.dino.config.hidden_size

        self.depth_hidden_channels = 128
        self.depth_out_channels = 64

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.depth_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.patch_h, self.patch_w)),
            nn.Conv2d(self.depth_hidden_channels, self.depth_out_channels, kernel_size=1),
        )

        fused_channels = self.rgb_feature_dim + self.depth_out_channels
        self.head_hidden_channels = 256
        self.prediction_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(fused_channels, self.head_hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.head_hidden_channels, 3),
        )

    def _resolve_device(self):
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device(self.device)

    def set_feature_extractor_v3(self, device):
        pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.processor.do_resize = False
        self.processor.do_center_crop = False
        self.processor.size = {"height": self.image_size[0], "width": self.image_size[1]}
        self.dino = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map=device,
        )
        for param in self.dino.parameters():
            param.requires_grad = False

    def _format_rgbd(self, rgbd: torch.Tensor) -> torch.Tensor:
        if rgbd.dim() == 3:
            if rgbd.shape[0] == 4:
                rgbd = rgbd.unsqueeze(0)
            elif rgbd.shape[-1] == 4:
                rgbd = rgbd.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError("Expected 4 channels for RGBD input")
        elif rgbd.dim() == 4:
            if rgbd.shape[1] != 4 and rgbd.shape[-1] == 4:
                rgbd = rgbd.permute(0, 3, 1, 2)
            elif rgbd.shape[1] != 4:
                raise ValueError("Expected RGBD tensor with 4 channels")
        else:
            raise ValueError("RGBD tensor must have 3 or 4 dimensions")
        return rgbd

    def encode_rgbd(self, rgbd: torch.Tensor) -> tuple[torch.Tensor, bool]:
        single_sample = rgbd.dim() == 3
        rgbd = self._format_rgbd(rgbd).float()
        runtime_device = self._resolve_device()
        rgbd = rgbd.to(runtime_device)

        rgb = rgbd[:, :3, :, :]
        depth = rgbd[:, 3:, :, :]
        if depth.shape[1] == 0:
            raise ValueError("Depth channel missing from RGBD input")

        rgb_images = [img.permute(1, 2, 0).detach().cpu().numpy() for img in rgb]
        inputs = self.processor(
            images=rgb_images,
            return_tensors="pt",
            do_rescale=True,
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
        outputs = self.dino(**inputs)
        register_tokens = getattr(self.dino.config, "num_register_tokens", 0)
        trim = 1 + register_tokens  # CLS + optional register tokens
        tokens = outputs.last_hidden_state[:, trim:, :]
        bsz, num_tokens, hidden = tokens.shape
        expected_tokens = self.patch_h * self.patch_w
        if num_tokens != expected_tokens:
            raise ValueError(
                f"Unexpected token count {num_tokens}; expected {expected_tokens} for input size {self.image_size}"
            )
        rgb_features = tokens.transpose(1, 2).reshape(bsz, hidden, self.patch_h, self.patch_w)
        depth_features = self.depth_projector(self.depth_encoder(depth))
        combined = torch.cat([rgb_features, depth_features], dim=1)
        return combined, single_sample

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        combined, single_sample = self.encode_rgbd(rgbd)
        preds = self.prediction_head(combined)
        if single_sample:
            preds = preds.squeeze(0)
        return preds


if __name__ == "__main__":
    device = "cuda"
    model = RGBDNN(device)
    model.to(device)

    from data import ImageRotationDataset
    import numpy as np

    def get_random_image():
        dataset = ImageRotationDataset("datasets/data_lungs_5", rgb=True, depth=True)
        idx = np.random.randint(len(dataset))
        print(f"Selected index: {idx}")
        rgbd, rotation = dataset[idx]

        return torch.tensor(rgbd, dtype=torch.float16), rotation
    
    image, rotation = get_random_image()

    from utils.images import show_image
    show_image(image[0].detach().cpu().numpy())
    
    print(f"Image shape: {image.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params} (trainable: {trainable_params})")
    
    output = model(image.to(device))
    print(f"Output shape: {output.shape}")
    print("Forward call working successfully!")