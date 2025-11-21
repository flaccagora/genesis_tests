import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

class RotationPredictor(nn.Module):
    """Rotation prediction model using DinoV3 backbone."""
    
    def __init__(self, backbone_dim=768, hidden_dim=512, use_rgbd=False):
        """
        Args:
            backbone_dim: output dim of DinoV3 backbone (usually 768)
            hidden_dim: hidden dimension of MLP head
            use_rgbd: if True, expects RGBD input (4 channels); if False, expects RGB (3 channels)
        """
        super().__init__()
        self.use_rgbd = use_rgbd
        
        # Try to load DinoV3, fallback to ResNet50 if not available
        try:
            pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
            # self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)

            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 768
            self.backbone_type = 'dinov3'
        except:
            print("DinoV3 not available, using ResNet50 instead")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 2048
            self.backbone_type = 'resnet50'
        
        # If using RGBD, add a depth preprocessing layer
        if use_rgbd:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Project to 3 channels to match RGB
            )
        
        # Prediction head: MLP
        self.head = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6)  # 6D rotation representation
        )
    
    def forward(self, x):
        """
        Args:
            x: input images (B, 3, H, W) for RGB or (B, 4, H, W) for RGBD
        Returns:
            6D rotation representation (B, 6)
        """
        if self.use_rgbd:
            # Split RGBD into RGB and Depth
            rgb = x[:, :3, :, :]  # (B, 3, H, W)
            depth = x[:, 3:4, :, :]  # (B, 1, H, W)
            
            # Encode depth to 3 channels and fuse with RGB
            depth_encoded = self.depth_encoder(depth)  # (B, 3, H, W)
            
            # Simple fusion: concatenate and average
            fused = torch.cat([rgb, depth_encoded], dim=1)  # (B, 6, H, W)
            x_input = fused.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)  # Average to 3 channels
        else:
            x_input = x
        
        # Extract features from backbone
        with torch.no_grad():
            if self.backbone_type == 'dinov3':
                features = self.backbone(x_input).last_hidden_state
            else:  # ResNet50
                features = self.backbone(x_input)
        
        # Global average pooling if needed
        if features.dim() > 2:
            features = torch.mean(features, dim=1)  # Global average pooling
        
        rot_6d = self.head(features)
        
        # Normalize to ensure valid rotation representation
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        
        return rot_6d

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 512

    model = RotationPredictor(hidden_dim=hidden_dim).to(device)

