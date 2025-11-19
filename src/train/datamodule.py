import os
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from data import ImageRotationDataset


DataLoaderLike = Union[DataLoader, List[DataLoader]]


class RotationDataModule(pl.LightningDataModule):
    """
    Thin LightningDataModule wrapper around the existing ImageRotationDataset.
    Expects directories that contain the paired image/.th rotation tensors
    produced by the current data pipeline.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        img_size: int = 224,
        shuffle: bool = True,
        rgb: bool = True,
        depth: bool = True
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.shuffle = shuffle
        self.rgb = rgb
        self.depth = depth
        transform_ops = [transforms.ToTensor()]
        if img_size:
            transform_ops.append(transforms.Resize((img_size, img_size)))
        self.transform = transforms.Compose(transform_ops)

        self.train_dataset: Optional[ImageRotationDataset] = None
        self.val_dataset: Optional[ImageRotationDataset] = None
        self.test_dataset: Optional[ImageRotationDataset] = None

    def _build_dataset(self, directory: Optional[str]) -> Optional[ImageRotationDataset]:
        if directory is None:
            return None
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Expected dataset directory at {directory}, but it was not found.")
        return ImageRotationDataset(directory, rgb=self.rgb, depth=self.depth, transform=self.transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(self.train_dir)
            self.val_dataset = self._build_dataset(self.val_dir)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._build_dataset(self.test_dir)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset has not been initialised. Did you forget to call setup('fit')?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoaderLike:
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoaderLike:
        if self.test_dataset is None:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

