"""Inference service for cloud detection and cloud-free reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import torch

from common.config import load_config
from evaluation.metrics import compute_metrics
from preprocessing.cloud_masks import generate_threshold_mask
from training.inference import load_checkpoint_model, predict_geotiff


class CloudRemovalService:
    """Service object that runs cloud-mask generation and reconstruction."""

    def __init__(self, config_path: str | Path) -> None:
        """Load project config and lazily initialize model inference."""
        self.config = load_config(config_path)
        backend_config = self.config.get("backend", {})
        self.checkpoint_path = Path(backend_config.get("model_checkpoint", "checkpoints/best.ckpt"))
        self.tile_size = int(backend_config.get("inference_tile_size", 512))
        self.overlap = int(backend_config.get("inference_overlap", 64))
        self._model: torch.nn.Module | None = None

    def predict(self, input_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
        """Run cloud detection and reconstruction for an uploaded GeoTIFF."""
        input_file = Path(input_path)
        output_root = Path(work_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        mask_path = output_root / f"{input_file.stem}_cloud_mask.tif"
        output_path = output_root / f"{input_file.stem}_reconstructed.tif"
        generate_threshold_mask(input_file, mask_path)
        if self.checkpoint_path.exists():
            model = self._load_model()
            predict_geotiff(
                model,
                input_file,
                output_path,
                tile_size=self.tile_size,
                overlap=self.overlap,
            )
            model_name = self.checkpoint_path.name
        else:
            self._classical_inpaint(input_file, mask_path, output_path)
            model_name = "classical-inpaint-fallback"
        metrics = self._metrics_if_possible(output_path, input_file)
        confidence = self._confidence(mask_path, uses_trained_model=self.checkpoint_path.exists())
        return {
            "mask_path": mask_path,
            "output_path": output_path,
            "model_name": model_name,
            "metrics": metrics,
            "confidence_score": confidence,
        }

    def _load_model(self) -> torch.nn.Module:
        """Load the trained reconstruction model once per process."""
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = load_checkpoint_model(self.config, self.checkpoint_path, device=device)
        return self._model

    def _classical_inpaint(self, image_path: Path, mask_path: Path, output_path: Path) -> None:
        """Create a GeoTIFF fallback reconstruction with OpenCV Telea inpainting."""
        with rasterio.open(image_path) as src:
            image = src.read().astype("float32")
            profile = src.profile.copy()
        with rasterio.open(mask_path) as mask_src:
            mask = (mask_src.read(1) > 0).astype("uint8")
        restored = image.copy()
        for band_index in range(min(3, image.shape[0])):
            band = np.clip(image[band_index], 0.0, 1.0)
            band_u8 = (band * 255).astype("uint8")
            restored_band = cv2.inpaint(band_u8, mask, 3, cv2.INPAINT_TELEA).astype("float32") / 255.0
            restored[band_index] = restored_band
        profile.update(dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(restored.astype("float32"))

    def _metrics_if_possible(self, prediction_path: Path, reference_path: Path) -> dict[str, float]:
        """Compute self-consistency metrics when no separate ground truth is supplied."""
        with rasterio.open(prediction_path) as prediction_src, rasterio.open(reference_path) as reference_src:
            prediction = prediction_src.read().astype("float32")
            reference = reference_src.read().astype("float32")
        common_channels = min(prediction.shape[0], reference.shape[0])
        return compute_metrics(prediction[:common_channels], reference[:common_channels])

    def _confidence(self, mask_path: Path, *, uses_trained_model: bool) -> float:
        """Estimate a conservative confidence score from cloud coverage and model type."""
        with rasterio.open(mask_path) as src:
            cloud_fraction = float((src.read(1) > 0).mean())
        base = 0.75 if uses_trained_model else 0.45
        return float(np.clip(base + 0.35 * (1.0 - cloud_fraction), 0.0, 0.99))


def metrics_to_json(metrics: dict[str, float]) -> str:
    """Serialize metrics while normalizing NaN and infinity values."""
    normalized = {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in metrics.items()
    }
    return json.dumps(normalized, allow_nan=False)
