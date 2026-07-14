"""Geospatial tiled inference utilities for trained reconstruction models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from models.factory import create_model


def load_checkpoint_model(config: dict[str, Any], checkpoint_path: str | Path, device: str | None = None) -> torch.nn.Module:
    """Load a configured Lightning model checkpoint for inference."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(target_device)


@torch.inference_mode()
def predict_geotiff(
    model: torch.nn.Module,
    stacked_input_path: str | Path,
    output_path: str | Path,
    *,
    tile_size: int = 512,
    overlap: int = 64,
) -> Path:
    """Run tiled model inference over a stacked multi-modal GeoTIFF."""
    src_path = Path(stacked_input_path)
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        out_channels = int(getattr(model, "hparams", {}).get("out_channels", 3))
        profile.update(count=out_channels, dtype="float32", nodata=0)
        accumulation = np.zeros((out_channels, src.height, src.width), dtype="float32")
        weights = np.zeros((1, src.height, src.width), dtype="float32")
        step = tile_size - overlap
        for row in range(0, src.height, step):
            for col in range(0, src.width, step):
                window = _bounded_window(row, col, tile_size, src.height, src.width)
                patch = src.read(window=window).astype("float32")
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                prediction = model(tensor).squeeze(0).detach().cpu().numpy()
                rows = slice(int(window.row_off), int(window.row_off + window.height))
                cols = slice(int(window.col_off), int(window.col_off + window.width))
                accumulation[:, rows, cols] += prediction[:, : int(window.height), : int(window.width)]
                weights[:, rows, cols] += 1.0
        reconstruction = accumulation / np.maximum(weights, 1e-6)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(reconstruction.astype("float32"))
    return dst_path


def _bounded_window(row: int, col: int, size: int, height: int, width: int) -> Window:
    """Create a rasterio window that stays inside image bounds."""
    bounded_row = min(row, max(0, height - size))
    bounded_col = min(col, max(0, width - size))
    window_height = min(size, height - bounded_row)
    window_width = min(size, width - bounded_col)
    return Window(bounded_col, bounded_row, window_width, window_height)
