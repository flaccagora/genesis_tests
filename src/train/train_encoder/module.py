from __future__ import annotations

from pathlib import Path
from typing import Optional

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

from models.NN_any import MeshAutoencoder


class MeshAutoencoderLightningModule(pl.LightningModule):
    """
    Lightning module wrapper for MeshAutoencoder.
    Trains the autoencoder to learn a low-dimensional latent representation
    of valid mesh deformations.
    """

    def __init__(
        self,
        n_vertices: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        encoder_type: str = "mlp",
        use_tnet: bool = False,
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

        self.save_hyperparameters()
        
        # Create model
        self.model = MeshAutoencoder(
            n_vertices=n_vertices,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            encoder_type=encoder_type,
            use_tnet=use_tnet,
        )

        if pretrained_path:
            state_dict = torch.load(Path(pretrained_path), map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        # Use MSE loss for reconstruction
        self.criterion = nn.MSELoss()
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

    def forward(self, mesh: torch.Tensor):
        return self.model(mesh)

    def _common_step(self, batch, batch_idx: int, stage: str):
        mesh = batch  # batch is just the mesh tensor
        reconstructed_mesh, latent_z = self.forward(mesh)
        
        # Reconstruction loss
        loss = self.criterion(reconstructed_mesh, mesh)
        
        # Log loss
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=True, on_epoch=True, sync_dist=True)
        
        # Log latent statistics during validation
        if stage == "val" and batch_idx == 0:
            self.log(f"{stage}_latent_mean", latent_z.mean(), on_epoch=True)
            self.log(f"{stage}_latent_std", latent_z.std(), on_epoch=True)
        
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
