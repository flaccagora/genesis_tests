import warnings
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
import logging
import torch.nn as nn
from transformers import AutoModel
from src.models.backbone.base import _BaseBackbone

logger = logging.getLogger(__name__)

class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vits14",
        source: str = "github",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logger.info(f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})")
            if backbone_kwargs:
                logger.info(f"DINO backbone kwargs: {backbone_kwargs}")
            
            self.backbone = torch.hub.load(
                repo_or_dir=repo_or_dir,
                model=dino_model,
                source=source,
                verbose=False,
                **backbone_kwargs,
            )
            
            # Log model properties after loading
            logger.info(f"Loaded DINO model - type: {type(self.backbone)}, "
                        f"embed_dim: {self.backbone.embed_dim}, "
                        f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}")


        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img):
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()

class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)

class DinoV2Backbone(_BaseBackbone):
    """DINOv2 backbone wrapper using the local Dino class from backbone/dino.py."""

    def __init__(
        self,
        input_size: int = 224,
        dino_model: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        normalize_images: bool = True,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        self.backbone = Dino(
            input_size=input_size,
            dino_model=dino_model,
            freeze_backbone=freeze_backbone,
            normalize_images=normalize_images,
        )
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = 14
        self.input_size = input_size

    def forward(self, x) -> torch.Tensor:
        """Returns (B, num_tokens, embed_dim) including CLS token."""
        return self.backbone(x)

class DinoV3Backbone(_BaseBackbone):
    """DINOv3 backbone wrapper using HuggingFace transformers."""

    def __init__(
        self,
        pretrained_model_name: str = "facebook/dinov3-vitB16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
    ):
        super().__init__(freeze_backbone=freeze_backbone)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.embed_dim = 768  # ViT-B output dimension
        self.patch_size = 16

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, x) -> torch.Tensor:
        """Returns (B, num_tokens, embed_dim) from last_hidden_state."""
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            output = self.backbone(x)
        return output.last_hidden_state


if __name__ == "__main__":
    model = Dino()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # Expected: (2, number_of_tokens, embed_dim)