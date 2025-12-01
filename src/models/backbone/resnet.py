import torch
import torch.nn as nn
from src.models.backbone.base import _BaseBackbone

class ResNet50Backbone(_BaseBackbone):
    """ResNet50 backbone wrapper that returns features as tokens."""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__(freeze_backbone=freeze_backbone)
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC and avgpool layers to get spatial features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 2048, H/32, W/32)
        self.embed_dim = 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, x) -> torch.Tensor:
        """Returns (B, num_tokens, embed_dim) by flattening spatial dimensions."""
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone(x)  # (B, 2048, H', W')
        B, C, H, W = features.shape
        # Flatten spatial dims to tokens: (B, H*W, C)
        tokens = features.flatten(2).permute(0, 2, 1)  # (B, num_tokens, embed_dim)
        return tokens

