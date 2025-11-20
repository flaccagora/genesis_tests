from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    StepLR,
    SequentialLR,
)

from models import DeformNet_v2, DeformNet_v3, DeformNet_v3_extractor, RGBDNN, RotationPredictor


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v2": DeformNet_v2,
    "v3": DeformNet_v3,
    "v3_extractor": DeformNet_v3_extractor,
    "RGBD": RGBDNN,
    "RotationPredictor": RotationPredictor
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
        use_lr_scheduler: bool = True,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 2,
        warmup_start_lr: float = 1e-6,
        cosine_final_lr: float = 1e-6,
        step_size: int = 10,
        gamma: float = 0.1,
        total_epochs: int = 10,
    ) -> None:
        super().__init__()

        if model_variant not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_variant='{model_variant}'. Options: {list(MODEL_REGISTRY)}")

        self.save_hyperparameters()
        model_cls = MODEL_REGISTRY[model_variant]
        self.model = model_cls()

        if pretrained_path:
            state_dict = torch.load(Path(pretrained_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        from loss.loss import geodesic_loss
        self.criterion = geodesic_loss
        self.lr = lr

        # Learning rate scheduler parameters
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.cosine_final_lr = cosine_final_lr
        self.step_size = step_size
        self.gamma = gamma
        self.total_epochs = total_epochs

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if not self.use_lr_scheduler:
            return optimizer
        
        # Create warmup scheduler if warmup_epochs > 0
        schedulers = []
        total_steps = self.total_epochs
        
        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.warmup_start_lr / self.lr,
                total_iters=self.warmup_epochs,
            )
            schedulers.append(warmup_scheduler)
        
        # Create main scheduler based on type
        main_epochs = self.total_epochs - self.warmup_epochs
        
        if self.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=self.cosine_final_lr,
            )
        elif self.scheduler_type == "linear":
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                total_iters=main_epochs,
            )
        elif self.scheduler_type == "exponential":
            main_scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        elif self.scheduler_type == "step":
            main_scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        schedulers.append(main_scheduler)
        
        # Use SequentialLR if we have warmup, otherwise just the main scheduler
        if self.warmup_epochs > 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=[self.warmup_epochs],
            )
        else:
            scheduler = main_scheduler
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        # Keep the wrapped model in sync with Lightning's device placement logic.
        if hasattr(self.model, "device"):
            self.model.device = str(self.device)

