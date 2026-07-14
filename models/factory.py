"""Model factory for CLI, backend, and tests."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from models.cgan import ConditionalGAN
from models.diffusion import DiffusionReconstructor
from models.transformer import PatchTransformerReconstructor
from models.unet import UNetReconstructor
from models.restormer_model import RestormerReconstructor

def create_model(config: dict[str, Any]) -> pl.LightningModule:
    """Instantiate a reconstruction model from a project config dictionary."""
    model_config = config.get("model", {})
    loss_weights = config.get("loss", {})
    common = {
        "in_channels": int(model_config.get("in_channels", 4)),
        "out_channels": int(model_config.get("out_channels", 4)),
        "base_channels": int(model_config.get("base_channels", 64)),
        "learning_rate": float(model_config.get("learning_rate", 2e-4)),
        "weight_decay": float(model_config.get("weight_decay", 1e-5)),
        "loss_weights": loss_weights,
    }
    name = str(model_config.get("name", "unet")).lower()
    if name in {"unet", "u-net", "baseline"}:
        return UNetReconstructor(**common)
    if name in {"cgan", "gan", "conditional_gan"}:
        return ConditionalGAN(**common)
    if name in {"diffusion", "ddpm"}:
        return DiffusionReconstructor(**common)
    if name in {"restormer", "restoration"}:
        return RestormerReconstructor(**common)
    if name in {"transformer", "vit"}:
        return PatchTransformerReconstructor(**common)
    raise ValueError(f"Unknown model architecture: {name}")
