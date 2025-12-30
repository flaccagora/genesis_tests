from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    StepLR,
    SequentialLR,
)

from models import RGB_RotationPredictor, RGBD_RotationPredictor, Dino_RGB_RotationPredictor, RGB_ActuationRotationPredictor
from loss.loss import GeodesicLoss, MSELoss
from utils.rotation import rot6d_to_rotmat

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "RGB_RotationPredictor": RGB_RotationPredictor,
    "RGBD_RotationPredictor": RGBD_RotationPredictor,
    "Dino_RGB_RotationPredictor": Dino_RGB_RotationPredictor,
    "RGB_ActuationRotationPredictor": RGB_ActuationRotationPredictor
}

CRITERION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mse": MSELoss,
    "geodesic": GeodesicLoss,
}


class DeformNetLightningModule(pl.LightningModule):
    """
    Thin LightningModule wrapper around the existing DeformNet architectures.
    Handles loss computation, logging, and optimizer configuration so that the
    rest of the codebase can call into a standard Lightning Trainer.
    """

    def __init__(
        self,
        model_cls: str = "RGB_RotationPredictor",
        backbone: str = "resnet",
        criterion: str = "mse",
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
        actu_weight: float = 1.0,
        rot_weight: float = 1.0,
        trans_weight: float = 1.0,
        p_init_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        if model_cls not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_cls='{model_cls}'. Options: {list(MODEL_REGISTRY)}")

        self.save_hyperparameters()
        model_cls_type = MODEL_REGISTRY[model_cls]
        self.model = model_cls_type(dino_model=backbone)

        if pretrained_path:
            state_dict = torch.load(Path(pretrained_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        self.criterion = CRITERION_REGISTRY[criterion]()
        self.lr = lr
        self.actu_weight = actu_weight
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        
        # Load initial particles
        self.register_buffer("p_init", None)
        if p_init_path and Path(p_init_path).exists():
            import numpy as np
            p_init = np.load(p_init_path)
            self.p_init = torch.from_numpy(p_init).float()
        elif p_init_path:
             print(f"Warning: p_init_path {p_init_path} provided but not found.")

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
        if len(batch) == 4:
            images, actu_targets, rot_targets, particles_targets = batch
            actu_preds, rot_preds, trans_preds = self.forward(images)
            
            loss_actu = F.mse_loss(actu_preds, actu_targets)
            
            rot_targets = self._prepare_targets(rot_targets).to(images.device)
            loss_rot = self.criterion(rot_preds, rot_targets)
            
            loss_trans = 0.0
            if self.p_init is not None:
                # Apply predicted rigid transform to initial particles
                # p_init: (N, 3) -> (B, N, 3)
                # rot_preds: (B, 6) -> (B, 3, 3)
                # trans_preds: (B, 3) -> (B, 1, 3)
                
                B = images.size(0)
                p_init_batch = self.p_init.unsqueeze(0).expand(B, -1, -1).to(images.device)
                
                rot_mat_preds = rot6d_to_rotmat(rot_preds) # (B, 3, 3)
                
                # R * P + t
                # (B, 3, 3) @ (B, N, 3)^T -> (B, 3, N) -> (B, N, 3)
                p_pred = torch.bmm(rot_mat_preds, p_init_batch.transpose(1, 2)).transpose(1, 2) + trans_preds.unsqueeze(1)
                
                loss_trans = F.mse_loss(p_pred, particles_targets)
            
            loss = self.actu_weight * loss_actu + self.rot_weight * loss_rot + self.trans_weight * loss_trans
            
            self.log(f"{stage}_loss_actu", loss_actu, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_loss_rot", loss_rot, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_loss_trans", loss_trans, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=True, on_epoch=True, sync_dist=True)
            return loss
        elif len(batch) == 3:
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

        # Watch model with wandb for gradient and parameter logging
        # if self.logger is not None:
        #     try:
        #         from pytorch_lightning.loggers import WandbLogger
        #         # Handle single logger or list of loggers
        #         loggers = self.logger if isinstance(self.logger, list) else [self.logger]
        #         for logger in loggers:
        #             if isinstance(logger, WandbLogger):
        #                 logger.watch(self.model, log="all", log_freq=100)
        #                 break
        #     except ImportError:
        #         raise ImportError("WandbLogger not found. ")

