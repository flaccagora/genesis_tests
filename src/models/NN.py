import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

class RotationPredictor(nn.Module):
    """Rotation prediction model using DinoV3 backbone."""
    
    def __init__(self, backbone_dim=768, hidden_dim=512):
        """
        Args:
            backbone_dim: output dim of DinoV3 backbone (usually 768)
            hidden_dim: hidden dimension of MLP head
        """
        super().__init__()
        
        # Try to load DinoV3, fallback to ResNet50 if not available
        try:
            pretrained_model_name = "facebook/dinov3-vitB16-pretrain-lvd1689m"
            # self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)

            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 768
        except:
            print("DinoV3 not available, using ResNet50 instead")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            backbone_output_dim = 2048
        
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
            x: input images (B, 3, H, W)
        Returns:
            6D rotation representation (B, 6)
        """
        with torch.no_grad():
            features = self.backbone(x)
        
        # Global average pooling if needed
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])
        
        rot_6d = self.head(features)
        
        # Normalize to ensure valid rotation representation
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        
        return rot_6d

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 512

    model = RotationPredictor(hidden_dim=hidden_dim).to(device)

