import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

class _BaseRotationPredictor(nn.Module):
    """Base rotation prediction model handling backbone and MLP head."""

    def __init__(self, backbone_dim=768, hidden_dim=512):
        super().__init__()

        # Try to load DinoV3, fallback to ResNet50 if not available
        try:
            pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
            # self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)

            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone_output_dim = 768
            self.backbone_type = 'dinov3'
        except Exception:
            print("DinoV3 not available, using ResNet50 instead")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone_output_dim = 2048
            self.backbone_type = 'resnet50'

        # Prediction head: MLP (outputs 6D rotation representation)
        self.head = nn.Sequential(
            nn.Linear(self.backbone_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),
        )

    def _extract_features(self, x_input):
        """Run backbone and do a small postprocessing to return a (B, D) feature tensor."""
        with torch.no_grad():
            if self.backbone_type == 'dinov3':
                features = self.backbone(x_input).last_hidden_state
            else:  # ResNet50
                features = self.backbone(x_input)

        # Global average pooling if needed
        if features.dim() > 2:
            features = torch.mean(features, dim=1)

        return features


class RGB_RotationPredictor(_BaseRotationPredictor):
    """Rotation predictor for RGB images (expects 3-channel input)."""

    def __init__(self, backbone_dim=768, hidden_dim=512):
        super().__init__(backbone_dim=backbone_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        """x: (B, 3, H, W) RGB images. Returns (B, 6) 6D rotation representation."""
        x_input = x
        features = self._extract_features(x_input)
        rot_6d = self.head(features)
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        return rot_6d


class RGBD_RotationPredictor(_BaseRotationPredictor):
    """Rotation predictor for RGBD images (expects 4-channel input: RGB + Depth).

    Depth is encoded with a small conv encoder and fused with RGB to produce a
    3-channel input compatible with DinoV3/ResNet backbones.
    """

    def __init__(self, backbone_dim=768, hidden_dim=512):
        super().__init__(backbone_dim=backbone_dim, hidden_dim=hidden_dim)

        # Depth preprocessing: project single-channel depth into 3 channels
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Project to 3 channels to match RGB
        )

    def forward(self, x):
        """x: (B, 4, H, W) where channels are [R,G,B,Depth]. Returns (B, 6)."""
        # Split RGBD into RGB and Depth
        rgb = x[:, :3, :, :]  # (B, 3, H, W)
        depth = x[:, 3:4, :, :]  # (B, 1, H, W)

        # Encode depth to 3 channels and fuse with RGB
        depth_encoded = self.depth_encoder(depth)  # (B, 3, H, W)

        # Simple fusion: average RGB and depth-encoding (keeps 3 channels)
        x_input = 0.5 * (rgb + depth_encoded)

        features = self._extract_features(x_input)
        rot_6d = self.head(features)
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)

        return rot_6d

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 512

    # instantiate an RGB-only predictor by default
    model = RGB_RotationPredictor(hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("RGB model output shape:", output.shape)

    model = RGBD_RotationPredictor(hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 4, 224, 224).to(device)
    output = model(dummy_input)
    print("RGB model output shape:", output.shape)


    # class DeformNet_v3_extractor(nn.Module):
    #     def __init__(self, device):
    #         super(DeformNet_v3_extractor, self).__init__()
    #         self.device = device
    #         self.set_feature_extractor_v3(device)

    #         self.cnn = nn.Sequential(

    #             # Block 1
    #             nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,201,768)
    #             nn.ReLU(),
    #             nn.MaxPool2d(2),                              # (B,16,100,384)

    #             # Block 2
    #             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,100,384)
    #             nn.ReLU(),
    #             nn.MaxPool2d(2),                              # (B,32,50,192)

    #             # Block 3
    #             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,50,192)
    #             nn.ReLU(),
    #             nn.MaxPool2d(2),                              # (B,64,25,96)

    #             # Block 4 — further reduce size
    #             nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B,128,25,96)
    #             nn.ReLU(),
    #             nn.MaxPool2d(2),                              # (B,128,12,48)

    #             # Block 5 — final reduction
    #             nn.Conv2d(128, 256, kernel_size=3, padding=1),# (B,256,12,48)
    #             nn.ReLU(),
    #             nn.MaxPool2d(2),                              # (B,256,6,24)
    #         )

    #         # Now flattened size: 256 * 6 * 24 = 36864
    #         self.fc = nn.Sequential(
    #             nn.Linear(256 * 6 * 24, 512),
    #             nn.ReLU(),
    #             nn.Linear(512, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 3)  # final output
    #         )
    #     def _resolve_device(self):
    #         param = next(self.parameters(), None)
    #         if param is not None:
    #             return param.device
    #         return torch.device(self.device)

    #     def set_feature_extractor_v3(self, device):
        
    #         pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
    #         self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    #         self.dino = AutoModel.from_pretrained(
    #             pretrained_model_name, 
    #             device_map=device,
    #             )

    #         for param in self.dino.parameters():
    #             param.requires_grad = False  # Freeze the feature extractor
            
            
    #     def forward(self, x):
    #         runtime_device = self._resolve_device()
    #         inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(runtime_device)
    #         x = self.dino(**inputs).last_hidden_state.unsqueeze(1)
    #         # print("EXPECTED 24 201 768 ",x.shape)
    #         x = self.cnn(x)
    #         x = x.view(x.size(0), -1)
    #         # print(x.shape)
    #         x = self.fc(x)
    #         return x
        