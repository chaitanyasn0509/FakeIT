"""PyTorch Lightning data module for multi-modal raster patches."""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
import torch

from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader, Dataset

INPUT_COLUMNS = [
    "cloudy"
]


class MultiModalPatchDataset(Dataset):
    """Dataset that stacks co-registered modalities and returns cloud-free targets."""

    def __init__(self, manifest: pd.DataFrame, *, augment: bool = False) -> None:
        """Create a patch dataset from a split-specific manifest DataFrame."""
        self.manifest = manifest.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Read one multi-modal patch sample from disk."""
        row = self.manifest.iloc[index]
        inputs = np.concatenate([self._read_array(row[column]) for column in INPUT_COLUMNS], axis=0)
        target = self._read_array(row["clear"])
        if self.augment:
            inputs, target = self._augment(inputs, target)
        return {
            "inputs": torch.tensor(inputs.copy(), dtype=torch.float32),
            "target": torch.tensor(target.copy(), dtype=torch.float32),
            "scene_id": str(row["scene_id"]),
        }

    def _read_array(self, path: str | Path) -> np.ndarray:
        """Read a 4-band Sentinel-2 patch and normalize to [0,1]."""

        with rasterio.open(path) as src:
            array = src.read().astype(np.float32)

        # Normalize Sentinel-2 reflectance
        array = np.clip(array, 0, 10000)
        array = array / 10000.0

        return np.nan_to_num(array)

    def _augment(self, inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply deterministic-safe flips to inputs and target together."""
        if np.random.rand() > 0.5:
            inputs = inputs[:, :, ::-1].copy()
            target = target[:, :, ::-1].copy()
        if np.random.rand() > 0.5:
            inputs = inputs[:, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
        return inputs, target


class ReconstructionDataModule(pl.LightningDataModule):
    """Lightning data module for train/validation/test raster patch splits."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Read data-loader settings from the project config."""
        super().__init__()
        self.config = config
        training_config = config.get("training", {})
        self.manifest_path = Path(config["data"]["manifest_path"])
        self.batch_size = int(training_config.get("batch_size", 2))
        self.num_workers = int(training_config.get("num_workers", 4))
        self.train_dataset: MultiModalPatchDataset | None = None
        self.val_dataset: MultiModalPatchDataset | None = None
        self.test_dataset: MultiModalPatchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load manifest rows and create split datasets."""
        manifest = pd.read_csv(self.manifest_path)
        self.train_dataset = MultiModalPatchDataset(
            manifest[manifest["split"] == "train"],
            augment=True,
        )
        self.val_dataset = MultiModalPatchDataset(manifest[manifest["split"] == "val"])
        self.test_dataset = MultiModalPatchDataset(manifest[manifest["split"] == "test"])

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""
        return DataLoader(
            self._require_dataset(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader."""
        return DataLoader(
            self._require_dataset(self.val_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader."""
        return DataLoader(
            self._require_dataset(self.test_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _require_dataset(self, dataset: MultiModalPatchDataset | None) -> MultiModalPatchDataset:
        """Return a dataset or raise a clear setup error."""
        if dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")
        return dataset
