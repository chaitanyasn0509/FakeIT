"""Cloud-mask generation strategies for LISS-IV imagery."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from skimage import morphology


def threshold_cloud_mask(
    image: np.ndarray,
    *,
    visible_bands: Iterable[int] = (1, 2, 3),
    brightness_quantile: float = 0.88,
    min_component_pixels: int = 64,
) -> np.ndarray:
    """Create a cloud mask from visible-band brightness and morphology."""
    zero_based = [band - 1 for band in visible_bands]
    visible = image[zero_based].astype("float32")
    brightness = visible.mean(axis=0)
    threshold = float(np.quantile(brightness, brightness_quantile))
    mask = brightness >= threshold
    mask = morphology.remove_small_objects(mask, min_size=min_component_pixels)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype("uint8")


def save_mask(mask: np.ndarray, reference_path: str | Path, output_path: str | Path) -> Path:
    """Save a binary mask as a georeferenced single-band GeoTIFF."""
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(mask.astype("uint8"), 1)
    return dst_path


def generate_threshold_mask(
    input_path: str | Path,
    output_path: str | Path,
    *,
    visible_bands: Iterable[int] = (1, 2, 3),
    brightness_quantile: float = 0.88,
    min_component_pixels: int = 64,
) -> Path:
    """Generate and save a threshold-based cloud mask for a raster."""
    with rasterio.open(input_path) as src:
        image = src.read().astype("float32")
    cloud_mask = threshold_cloud_mask(
        image,
        visible_bands=visible_bands,
        brightness_quantile=brightness_quantile,
        min_component_pixels=min_component_pixels,
    )
    return save_mask(cloud_mask, input_path, output_path)


def run_fmask(input_path: str | Path, output_path: str | Path, executable: str = "fmask") -> Path:
    """Run an installed FMask executable and write its cloud-mask output."""
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, str(input_path), str(dst_path)]
    subprocess.run(command, check=True)
    return dst_path


class TorchCloudSegmenter:
    """Deep-learning cloud segmenter for checkpointed PyTorch models."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        """Load a TorchScript or state-dict checkpoint for inference."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, torch.jit.ScriptModule):
            self.model = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model = checkpoint["model"]
        else:
            raise ValueError("Deep cloud mask checkpoint must contain a serialized model.")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict a binary cloud mask from a band-first image array."""
        tensor = torch.from_numpy(image.astype("float32")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        return (probabilities >= threshold).astype("uint8")

    def predict_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        """Predict and save a cloud mask for a GeoTIFF file."""
        with rasterio.open(input_path) as src:
            image = src.read().astype("float32")
        return save_mask(self.predict(image), input_path, output_path)
