"""Tests for reconstruction metrics."""

from __future__ import annotations

import numpy as np

from evaluation.metrics import compute_metrics, sam


def test_compute_metrics_identical_arrays() -> None:
    """Identical arrays produce perfect or near-perfect quality metrics."""
    target = np.ones((3, 8, 8), dtype="float32") * 0.5
    metrics = compute_metrics(target, target)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["ssim"] > 0.99


def test_sam_is_small_for_identical_spectra() -> None:
    """SAM remains near zero for identical spectral vectors."""
    target = np.random.default_rng(42).random((3, 8, 8), dtype=np.float32)
    assert sam(target, target) < 0.01
