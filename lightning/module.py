from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models import DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v2": DeformNet_v2,
    "v3": DeformNet_v3,
    "v3_extractor": DeformNet_v3_extractor,
}


class DeformNetLightningModule(pl.LightningModule):
    """
    Thin LightningModule wrapper around the existing DeformNet architectures.
    Handles loss computation, logging, and optimizer configuration so that the
    rest of the codebase can call into a standard Lightning Trainer.
    """

    def __init__(
        self,
        model_variant: str = "v3",
        lr: float = 1e-3,
        compile_model: bool = False,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if model_variant not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_variant='{model_variant}'. Options: {list(MODEL_REGISTRY)}")

        self.save_hyperparameters()
        model_cls = MODEL_REGISTRY[model_variant]
        self.model = model_cls(device="cpu")

        if pretrained_path:
            state_dict = torch.load(Path(pretrained_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _prepare_targets(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        if rotation_matrices.ndim == 3 and rotation_matrices.size(1) == 1:
            rotation_matrices = rotation_matrices.squeeze(1)
        return rotation_matrices.float()

    def _common_step(self, batch, batch_idx: int, stage: str):
        images, rotation_matrices = batch
        targets = self._prepare_targets(rotation_matrices).to(images.device)
        preds = self.forward(images)
        loss = self.criterion(preds, targets)
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_fit_start(self) -> None:
        # Keep the wrapped model in sync with Lightning's device placement logic.
        if hasattr(self.model, "device"):
            self.model.device = str(self.device)

