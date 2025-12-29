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

from models.NN_any import MeshAutoencoder, ImageToMeshPredictor


class ImageToMeshLightningModule(pl.LightningModule):
    """
    Lightning module wrapper for ImageToMeshPredictor.
    Trains the image encoder to predict mesh deformations from RGB images.
    The decoder is loaded from a pretrained autoencoder and frozen.
    """

    def __init__(
        self,
        n_vertices: int,
        latent_dim: int = 32,
        pretrained_decoder_path: Optional[str] = None,
        pretrained_resnet: bool = True,
        lr: float = 1e-3,
        compile_model: bool = False,
        use_lr_scheduler: bool = True,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 2,
        warmup_start_lr: float = 1e-6,
        cosine_final_lr: float = 1e-6,
        step_size: int = 10,
        gamma: float = 0.1,
        total_epochs: int = 10,
        freeze_decoder: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        
        # Load pretrained decoder if path provided
        if pretrained_decoder_path:
            print(f"Loading pretrained decoder from: {pretrained_decoder_path}")
            checkpoint = torch.load(Path(pretrained_decoder_path), map_location="cpu")
            
            # Extract model state dict (handle Lightning checkpoint format)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Remove 'model.' prefix if present
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Create a temporary autoencoder to load the decoder
            temp_autoencoder = MeshAutoencoder(
                n_vertices=n_vertices,
                latent_dim=latent_dim,
            )
            temp_autoencoder.load_state_dict(state_dict, strict=False)
            mesh_decoder = temp_autoencoder.decoder
            print("✓ Pretrained decoder loaded successfully")
        else:
            # Create a new decoder (not recommended, should use pretrained)
            print("WARNING: No pretrained decoder path provided. Creating new decoder.")
            temp_autoencoder = MeshAutoencoder(
                n_vertices=n_vertices,
                latent_dim=latent_dim,
            )
            mesh_decoder = temp_autoencoder.decoder
        
        # Create the image-to-mesh predictor
        self.model = ImageToMeshPredictor(
            mesh_decoder=mesh_decoder,
            n_vertices=n_vertices,
            latent_dim=latent_dim,
            pretrained_resnet=pretrained_resnet,
        )
        
        # Freeze decoder if requested
        if freeze_decoder:
            for param in self.model.mesh_decoder.parameters():
                param.requires_grad = False
            print("✓ Decoder frozen")

        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        # Use MSE loss for mesh reconstruction
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

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def _common_step(self, batch, batch_idx: int, stage: str):
        # Unpack batch
        rgb = batch["rgb"]
        target_particles = batch["particles"]
        
        # Forward pass: image -> predicted mesh
        predicted_mesh, predicted_z = self.forward(rgb)
        
        # Compute loss between predicted and target particles
        loss = self.criterion(predicted_mesh, target_particles)
        
        # Log loss
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"), on_step=True, on_epoch=True, sync_dist=True)
        
        # Log latent statistics during validation
        if stage == "val" and batch_idx == 0:
            self.log(f"{stage}_latent_mean", predicted_z.mean(), on_epoch=True)
            self.log(f"{stage}_latent_std", predicted_z.std(), on_epoch=True)
        
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
