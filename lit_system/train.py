from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger as PLLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lit_system import DeformNetLightningModule, RotationDataModule


LoggerReturn = Union[PLLogger, List[PLLogger], bool]


def build_loggers(config: Dict[str, Any]) -> LoggerReturn:
    loggers: List[PLLogger] = []

    if config["use_wandb"]:
        wandb_logger = WandbLogger(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            save_dir=config["default_root_dir"],
            name=config["experiment_name"],
            log_model=config["wandb_log_model"],
            offline=config["wandb_offline"],
            group=config["wandb_group"],
            tags=config["wandb_tags"],
        )
        loggers.append(wandb_logger)

    if config["use_tensorboard_logger"]:
        tensorboard_logger = TensorBoardLogger(
            save_dir=config["default_root_dir"],
            name=config["experiment_name"],
        )
        loggers.append(tensorboard_logger)

    if not loggers:
        return False
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def build_trainer(config: Dict[str, Any], monitor_val: bool) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            filename=config["checkpoint_name"],
            monitor="val_loss" if monitor_val else "train_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        default_root_dir=config["default_root_dir"],
        logger=build_loggers(config),
        callbacks=callbacks,
        limit_train_batches=config["limit_train_batches"],
        limit_val_batches=config["limit_val_batches"],
        limit_test_batches=config["limit_test_batches"],
    )

    return trainer


def run(config: Dict[str, Any]) -> None:
    data_module = RotationDataModule(
        train_dir=config["train_dir"],
        val_dir=config["val_dir"],
        test_dir=config["test_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        img_size=config["img_size"],
        shuffle=config["shuffle"],
    )

    lightning_module = DeformNetLightningModule(
        model_variant=config["model_variant"],
        lr=config["learning_rate"],
        compile_model=config["compile_model"],
        pretrained_path=config["pretrained_path"],
    )

    monitor_val = config["val_dir"] is not None
    trainer = build_trainer(config, monitor_val=monitor_val)

    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=config["resume_from"],
    )

    if config["test_dir"] is not None:
        trainer.test(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    # -----------------------------------------------------------------------------
    # Defaults (can be overridden via config files/flags with utils/configurator.py)
    # Data
    train_dir = "dataset"
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    batch_size = 32
    num_workers = 0
    img_size = 224
    shuffle = True

    # Model
    model_variant = "v3"
    learning_rate = 1e-3
    compile_model = False
    pretrained_path: Optional[str] = None

    # Trainer
    max_epochs = 10
    accelerator: Union[str, int] = "auto"
    devices: Union[str, List[int]] = "auto"
    precision = "bf16-mixed"
    default_root_dir = "lightning_logs"
    experiment_name = "deformnet"
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    checkpoint_name = "deformnet-{epoch:02d}-{val_loss:.4f}"
    resume_from: Optional[str] = None

    # Logging
    use_wandb = True
    use_tensorboard_logger = False
    wandb_project = "deformnet"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: List[str] = []
    wandb_log_model: Union[str, bool] = "all"
    wandb_offline = False

    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str, list, tuple, dict, type(None)))
    ]
    exec(open("utils/configurator.py").read())  # type: ignore[misc]  # noqa: S102
    config: Dict[str, Any] = {k: globals()[k] for k in config_keys}

    run(config)

