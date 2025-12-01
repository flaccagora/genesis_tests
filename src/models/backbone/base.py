import torch.nn as nn
import torch

class _BaseBackbone(nn.Module):
    """Abstract base class for backbone wrappers that return token features."""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.embed_dim: int = 0  # To be set by subclasses

    def forward(self, x) -> torch.Tensor:
        """Returns (B, num_tokens, embed_dim) token features."""
        raise NotImplementedError

