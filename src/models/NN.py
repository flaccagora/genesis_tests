import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.models.backbone.dino import Dino, DinoV2Backbone, DinoV3Backbone
from src.models.backbone.resnet import ResNet50Backbone

class _BaseRotationPredictor(nn.Module):
    """Base rotation prediction model with configurable backbone and MLP head.
    
    Args:
        backbone_type: One of 'dinov2', 'dinov3', or 'resnet'
        hidden_dim: Hidden dimension for the MLP head
        input_size: Input image size (used for dinov2)
        dino_model: DINOv2 model variant (e.g., 'dinov2_vitb14', 'dinov2_vits14')
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        backbone_type: str = 'dinov2',
        hidden_dim: int = 512,
        input_size: int = 224,
        dino_model: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone_type = backbone_type

        # Initialize backbone based on type
        if backbone_type == 'dinov2':
            self.backbone = DinoV2Backbone(
                input_size=input_size,
                dino_model=dino_model,
                freeze_backbone=freeze_backbone,
                normalize_images=False,
            )
        elif backbone_type == 'dinov3':
            self.backbone = DinoV3Backbone(freeze_backbone=freeze_backbone)
        elif backbone_type == 'resnet':
            self.backbone = ResNet50Backbone(freeze_backbone=freeze_backbone)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Choose from 'dinov2', 'dinov3', 'resnet'.")

        self.backbone_output_dim = self.backbone.embed_dim


class RGB_RotationPredictor(_BaseRotationPredictor):
    """Rotation predictor for RGB images (expects 3-channel input)."""

    def __init__(
        self,
        backbone_type: str = 'dinov2',
        hidden_dim: int = 512,
        input_size: int = 224,
        dino_model: str = "dinov2_vits14",
        freeze_backbone: bool = True,
    ):
        super().__init__(
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
            input_size=input_size,
            dino_model=dino_model,
            freeze_backbone=freeze_backbone,
        )
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

    def forward(self, x):
        """x: (B, 3, H, W) RGB images. Returns (B, 6) 6D rotation representation."""
        tokens = self.backbone(x)  # (B, num_tokens, embed_dim)
        features = tokens.mean(dim=1)  # (B, embed_dim)
        rot_6d = self.head(features)
        rot_6d = F.normalize(rot_6d.reshape(-1, 2, 3), dim=-1).reshape(-1, 6)
        return rot_6d


class RGBD_RotationPredictor(_BaseRotationPredictor):
    """Rotation predictor for RGBD images (expects 4-channel input: RGB + Depth).

    Depth is encoded with a small conv encoder and fused with RGB to produce a
    3-channel input compatible with all backbones.
    """

    def __init__(
        self,
        backbone_type: str = 'dinov2',
        hidden_dim: int = 512,
        input_size: int = 224,
        dino_model: str = "dinov2_vitb14",
        freeze_backbone: bool = True,
    ):
        super().__init__(
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
            input_size=input_size,
            dino_model=dino_model,
            freeze_backbone=freeze_backbone,
        )

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

        tokens = self.backbone(x_input)  # (B, num_tokens, embed_dim)
        features = tokens.mean(dim=1)  # (B, embed_dim)
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
            normalize_images=False,
        )
        self.backbone_output_dim = self.backbone.embed_dim  # typically 768 for vitb14

        # Compute number of patch tokens from input size and patch size (14 for dinov2)
        patch_size = 14
        num_patches = (input_size // patch_size) ** 2
        # +1 for CLS token from backbone, +1 for our query token
        self.num_tokens = num_patches + 1 + 1

        # Project backbone features to hidden_dim if different
        self.input_proj = nn.Linear(self.backbone_output_dim, hidden_dim) if self.backbone_output_dim != hidden_dim else nn.Identity()

        # Learnable query token for aggregating information
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Learnable positional embeddings for all tokens (query + CLS + patches)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)

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

        # Add positional embeddings
        tokens = tokens + self.pos_embedding

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

        # Compute number of patch tokens from input size and patch size (14 for dinov2)
        patch_size = 14
        num_patches = (input_size // patch_size) ** 2
        # +1 for CLS token from backbone, +1 for our query token
        self.num_tokens = num_patches + 1 + 1

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

        # Learnable positional embeddings for all tokens (query + CLS + patches)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)

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

        # Add positional embeddings
        tokens = tokens + self.pos_embedding

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

    # Test RGB predictor with DINOv2 backbone (default)
    print("Testing RGB_RotationPredictor with DINOv2 backbone...")
    model = RGB_RotationPredictor(backbone_type='dinov2', hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("RGB DINOv2 model output shape:", output.shape)

    # Test RGBD predictor with DINOv2 backbone
    print("\nTesting RGBD_RotationPredictor with DINOv2 backbone...")
    model = RGBD_RotationPredictor(backbone_type='dinov2', hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 4, 224, 224).to(device)
    output = model(dummy_input)
    print("RGBD DINOv2 model output shape:", output.shape)

    # Test RGB predictor with ResNet backbone
    print("\nTesting RGB_RotationPredictor with ResNet backbone...")
    model = RGB_RotationPredictor(backbone_type='resnet', hidden_dim=hidden_dim).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("RGB ResNet model output shape:", output.shape)

    # Test Dino-based RGB predictor (Transformer head)
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

    # Test Dino-based RGBD predictor (Transformer head)
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
        