"""Image-quality and spectral metrics for reconstructed satellite imagery."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity


@dataclass(slots=True)
class MetricResult:
    """Named metric value with units and directionality metadata."""

    name: str
    value: float
    higher_is_better: bool


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio in decibels."""
    mse = np.mean((prediction - target) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20 * math.log10(data_range / math.sqrt(mse)))


def ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute mean structural similarity over band-first imagery."""
    return float(
        structural_similarity(
            target.transpose(1, 2, 0),
            prediction.transpose(1, 2, 0),
            channel_axis=2,
            data_range=data_range,
        )
    )


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def sam(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute spectral angle mapper in radians."""
    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    numerator = np.sum(pred * truth, axis=0)
    denominator = np.linalg.norm(pred, axis=0) * np.linalg.norm(truth, axis=0)
    cosine = np.clip(numerator / np.maximum(denominator, 1e-8), -1 + 1e-6, 1 - 1e-6)
    return float(np.mean(np.arccos(cosine)))


def ergas(prediction: np.ndarray, target: np.ndarray, resolution_ratio: float = 1.0) -> float:
    """Compute ERGAS global relative dimensionless synthesis error."""
    band_rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=(1, 2)))
    band_mean = np.mean(target, axis=(1, 2))
    value = 100.0 / resolution_ratio * np.sqrt(np.mean((band_rmse / np.maximum(band_mean, 1e-6)) ** 2))
    return float(value)


def uiqi(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute Universal Image Quality Index averaged over bands."""
    values = []
    for band_index in range(prediction.shape[0]):
        x = prediction[band_index]
        y = target[band_index]
        mean_x = x.mean()
        mean_y = y.mean()
        var_x = x.var()
        var_y = y.var()
        covariance = ((x - mean_x) * (y - mean_y)).mean()
        denominator = (var_x + var_y) * (mean_x**2 + mean_y**2)
        values.append((4 * covariance * mean_x * mean_y) / max(float(denominator), 1e-8))
    return float(np.mean(values))


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute the standard metric suite for one prediction-target pair."""
    prediction = np.clip(prediction.astype("float32"), 0.0, 1.0)
    target = np.clip(target.astype("float32"), 0.0, 1.0)
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
        "sam": sam(prediction, target),
        "ergas": ergas(prediction, target),
        "uiqi": uiqi(prediction, target),
    }


class DeepMetricAccumulator:
    """Accumulate LPIPS and FID over an evaluation split."""

    def __init__(self, device: str = "cpu") -> None:
        """Create TorchMetrics objects for perceptual and distribution metrics."""
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.torch = torch
        self.device = torch.device(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.lpips_values: list[float] = []

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Update LPIPS and FID state with one prediction-target pair."""
        prediction_tensor = self._float_tensor(prediction)
        target_tensor = self._float_tensor(target)
        lpips_value = self.lpips(prediction_tensor * 2 - 1, target_tensor * 2 - 1)
        self.lpips_values.append(float(lpips_value.detach().cpu()))
        self.fid.update(self._uint8_tensor(target), real=True)
        self.fid.update(self._uint8_tensor(prediction), real=False)

    def compute(self) -> dict[str, float]:
        """Return split-level LPIPS and FID values."""
        lpips_value = float(np.mean(self.lpips_values)) if self.lpips_values else float("nan")
        fid_value = float(self.fid.compute().detach().cpu()) if self.lpips_values else float("nan")
        return {"lpips": lpips_value, "fid": fid_value}

    def _float_tensor(self, image: np.ndarray) -> object:
        """Convert a band-first array to a normalized three-channel tensor."""
        image = _to_three_channels(np.clip(image, 0.0, 1.0))
        return self.torch.from_numpy(image).unsqueeze(0).float().to(self.device)

    def _uint8_tensor(self, image: np.ndarray) -> object:
        """Convert a band-first array to a uint8 three-channel tensor."""
        image = (_to_three_channels(np.clip(image, 0.0, 1.0)) * 255).astype("uint8")
        return self.torch.from_numpy(image).unsqueeze(0).to(self.device)


def _to_three_channels(image: np.ndarray) -> np.ndarray:
    """Select or repeat bands to produce a three-channel band-first array."""
    if image.shape[0] == 3:
        return image
    if image.shape[0] > 3:
        return image[:3]
    return np.repeat(image, 3, axis=0)[:3]


def summarize_metric_table(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dictionaries into a single summary dictionary."""
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "scene_id"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
