import os
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import ANYDataset


DataLoaderLike = Union[DataLoader, List[DataLoader]]


class ANYDataModule(pl.LightningDataModule):
    """
    LightningDataModule for ANYDataset.
    Loads multi-modal data: RGB, depth, particles, rotation, actuator.
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
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.shuffle = shuffle
        
        # Transform for RGB images
        transform_ops = [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        self.transform = transforms.Compose(transform_ops)

        self.train_dataset: Optional[ANYDataset] = None
        self.val_dataset: Optional[ANYDataset] = None
        self.test_dataset: Optional[ANYDataset] = None

    def _build_dataset(self, directory: Optional[str]) -> Optional[ANYDataset]:
        if directory is None:
            return None
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Expected dataset directory at {directory}, but it was not found.")
        return ANYDataset(directory, transform=self.transform)

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
            persistent_workers=self.num_workers > 0,
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
