import os
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


DataLoaderLike = Union[DataLoader, List[DataLoader]]


class MeshDataset(Dataset):
    """
    Simple dataset for loading mesh point clouds from .npy files.
    Each file contains mesh vertices of shape [N_vertices, 3].
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Find all .npy files in the directory."""
        samples = []
        for fname in sorted(os.listdir(self.root_dir)):
            if fname.endswith('.npy'):
                samples.append(os.path.join(self.root_dir, fname))
        
        if not samples:
            raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mesh = np.load(self.samples[idx])
        return torch.from_numpy(mesh).float()


class MeshDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading mesh point clouds.
    Expects directories containing .npy files with mesh vertices.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset: Optional[MeshDataset] = None
        self.val_dataset: Optional[MeshDataset] = None
        self.test_dataset: Optional[MeshDataset] = None

    def _build_dataset(self, directory: Optional[str]) -> Optional[MeshDataset]:
        if directory is None:
            return None
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Expected dataset directory at {directory}, but it was not found.")
        return MeshDataset(directory)

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
