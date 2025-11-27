import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel

from src.models.backbone.dino import Dino

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


class Dino_RGB_RotationPredictor(nn.Module):
    """Rotation predictor for RGB images using the local Dino backbone (dinov2).

    Uses the Dino class from backbone/dino.py which loads dinov2 from torch hub.
    Expects 3-channel RGB input. Uses a vanilla Transformer encoder as the prediction head.
    """

    def __init__(
        self,
        hidden_dim: int = 392,
        input_size: int = 392,
        dino_model: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initialize local Dino backbone (dinov2)
        self.backbone = Dino(
            input_size=input_size,
            dino_model=dino_model,
            freeze_backbone=freeze_backbone,
        )
        self.backbone_output_dim = self.backbone.embed_dim  # typically 768 for vitb14

        # Project backbone features to hidden_dim if different
        self.input_proj = nn.Linear(self.backbone_output_dim, hidden_dim) if self.backbone_output_dim != hidden_dim else nn.Identity()

        # Learnable query token for aggregating information
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Vanilla Transformer encoder as prediction head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final projection to 6D rotation representation
        self.output_proj = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        """x: (B, 3, H, W) RGB images. Returns (B, 6) 6D rotation representation."""
        B = x.shape[0]

        # Dino backbone returns (B, num_tokens, embed_dim)
        tokens = self.backbone(x)

        # Project to hidden_dim
        tokens = self.input_proj(tokens)  # (B, num_tokens, hidden_dim)

        # Prepend learnable query token
        query_tokens = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        tokens = torch.cat([query_tokens, tokens], dim=1)  # (B, 1 + num_tokens, hidden_dim)

        # Pass through Transformer encoder
        tokens = self.transformer_head(tokens)  # (B, 1 + num_tokens, hidden_dim)

        # Use the query token output as aggregated feature
        aggregated = tokens[:, 0, :]  # (B, hidden_dim)

        # Project to 6D rotation
        rot_6d = self.output_proj(aggregated)
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        return rot_6d


class Dino_RGBD_RotationPredictor(nn.Module):
    """Rotation predictor for RGBD images using the local Dino backbone (dinov2).

    Uses the Dino class from backbone/dino.py. Depth is encoded with a small
    conv encoder and fused with RGB before passing to the Dino backbone.
    Expects 4-channel input: [R, G, B, Depth].
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        input_size: int = 224,
        dino_model: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initialize local Dino backbone (dinov2)
        self.backbone = Dino(
            input_size=input_size,
            dino_model=dino_model,
            freeze_backbone=freeze_backbone,
        )
        self.backbone_output_dim = self.backbone.embed_dim

        # Depth preprocessing: project single-channel depth into 3 channels
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Project to 3 channels
        )

        # Project backbone features to hidden_dim if different
        self.input_proj = nn.Linear(self.backbone_output_dim, hidden_dim) if self.backbone_output_dim != hidden_dim else nn.Identity()

        # Learnable query token for aggregating information
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Vanilla Transformer encoder as prediction head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final projection to 6D rotation representation
        self.output_proj = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        """x: (B, 4, H, W) where channels are [R,G,B,Depth]. Returns (B, 6)."""
        B = x.shape[0]

        # Split RGBD into RGB and Depth
        rgb = x[:, :3, :, :]  # (B, 3, H, W)
        depth = x[:, 3:4, :, :]  # (B, 1, H, W)

        # Encode depth to 3 channels and fuse with RGB
        depth_encoded = self.depth_encoder(depth)  # (B, 3, H, W)

        # Simple fusion: average RGB and depth-encoding (keeps 3 channels)
        x_input = 0.5 * (rgb + depth_encoded)

        # Dino backbone returns (B, num_tokens, embed_dim)
        tokens = self.backbone(x_input)

        # Project to hidden_dim
        tokens = self.input_proj(tokens)  # (B, num_tokens, hidden_dim)

        # Prepend learnable query token
        query_tokens = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        tokens = torch.cat([query_tokens, tokens], dim=1)  # (B, 1 + num_tokens, hidden_dim)

        # Pass through Transformer encoder
        tokens = self.transformer_head(tokens)  # (B, 1 + num_tokens, hidden_dim)

        # Use the query token output as aggregated feature
        aggregated = tokens[:, 0, :]  # (B, hidden_dim)

        # Project to 6D rotation
        rot_6d = self.output_proj(aggregated)
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
    print("RGBD model output shape:", output.shape)

    # Test Dino-based RGB predictor
    print("\nTesting Dino_RGB_RotationPredictor...")
    model = Dino_RGB_RotationPredictor(hidden_dim=768).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # backbone parameters
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"backbone parameters (frozen): {backbone_params}")
    # print model structure
    print(model)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("Dino RGB model output shape:", output.shape)

    # Test Dino-based RGBD predictor
    print("\nTesting Dino_RGBD_RotationPredictor...")
    model = Dino_RGBD_RotationPredictor(hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 4, 224, 224).to(device)
    output = model(dummy_input)
    print("Dino RGBD model output shape:", output.shape)


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
        