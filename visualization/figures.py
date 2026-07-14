"""Generate qualitative comparison figures for reconstruction reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def read_rgb(path: str | Path, bands: tuple[int, int, int] = (1, 2, 3)) -> np.ndarray:
    """Read and normalize selected raster bands as an RGB image."""
    with rasterio.open(path) as src:
        image = src.read(bands).astype("float32")
    image = np.clip(image, 0.0, 1.0)
    return np.moveaxis(image, 0, -1)


def read_single_band(path: str | Path) -> np.ndarray:
    """Read the first band of a raster for mask visualization."""
    with rasterio.open(path) as src:
        return src.read(1)


def save_comparison_figure(
    cloudy_path: str | Path,
    mask_path: str | Path,
    target_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Save a figure containing cloudy input, mask, target, prediction, and difference."""
    cloudy = read_rgb(cloudy_path)
    mask = read_single_band(mask_path)
    target = read_rgb(target_path)
    prediction = read_rgb(prediction_path)
    difference = np.abs(prediction - target).mean(axis=2)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    panels = [
        ("Cloudy input", cloudy, "viridis"),
        ("Cloud mask", mask, "gray"),
        ("Ground truth", target, "viridis"),
        ("Prediction", prediction, "viridis"),
        ("Difference", difference, "magma"),
    ]
    for axis, (title, image, cmap) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
