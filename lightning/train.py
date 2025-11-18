import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from lightning import DeformNetLightningModule, RotationDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeformNet variants using PyTorch Lightning.")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training samples.")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional directory with validation samples.")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional directory with test samples.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for all dataloaders.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloaders.")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for transforms.")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle training data each epoch.")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffling for training data.")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--model_variant", type=str, default="v3", choices=["v2", "v3", "v3_extractor"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the model.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Optional checkpoint to load before training.")

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto", help="Devices argument passed to Lightning Trainer.")
    parser.add_argument("--precision", default="bf16-mixed", help="Precision policy for Trainer.")
    parser.add_argument("--default_root_dir", type=str, default="lightning_logs")
    parser.add_argument("--experiment_name", type=str, default="deformnet")
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--limit_val_batches", default=1.0, type=float)
    parser.add_argument("--limit_test_batches", default=1.0, type=float)
    parser.add_argument("--checkpoint_name", type=str, default="deformnet-{epoch:02d}-{val_loss:.4f}")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from a Lightning checkpoint.")

    return parser.parse_args()


def build_trainer(args: argparse.Namespace, monitor_val: bool) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            filename=args.checkpoint_name,
            monitor="val_loss" if monitor_val else "train_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = WandbLogger(save_dir=args.default_root_dir, name=args.experiment_name)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    return trainer


def main():
    args = parse_args()

    data_module = RotationDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        shuffle=args.shuffle,
    )

    lightning_module = DeformNetLightningModule(
        model_variant=args.model_variant,
        lr=args.learning_rate,
        compile_model=args.compile,
        pretrained_path=args.pretrained_path,
    )

    monitor_val = args.val_dir is not None
    trainer = build_trainer(args, monitor_val=monitor_val)

    trainer.fit(
        lightning_module,
        datamodule=data_module,
        ckpt_path=args.resume_from,
    )

    if args.test_dir is not None:
        trainer.test(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()

