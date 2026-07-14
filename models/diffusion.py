"""Conditional denoising diffusion model for LISS-IV reconstruction."""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from models.unet import UNetReconstructor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        """Initialize embedding with MLP projection."""
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps into a continuous vector representation."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class DiffusionReconstructor(pl.LightningModule):
    """DDPM-style model conditioned on cloudy LISS-IV and auxiliary modalities."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        learning_rate: float,
        weight_decay: float,
        loss_weights: dict[str, float],
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
    ) -> None:
        """Create the noise-prediction network and DDPM scheduler."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.out_channels = out_channels
        self.num_inference_steps = num_inference_steps
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        self.noise_net = UNetReconstructor(
            in_channels + out_channels,
            out_channels,
            base_channels,
            learning_rate,
            weight_decay,
            {name: value for name, value in loss_weights.items() if name == "l1"},
            output_activation="identity",
        )
        net_channels = in_channels + out_channels
        self.time_embed = SinusoidalTimestepEmbedding(base_channels)
        self.time_proj = nn.Linear(base_channels, net_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sample a cloud-free reconstruction through iterative denoising."""
        sample = torch.randn(
            inputs.shape[0],
            self.out_channels,
            inputs.shape[2],
            inputs.shape[3],
            device=inputs.device,
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=inputs.device)
        for timestep in self.scheduler.timesteps:
            net_in = torch.cat([inputs, sample], dim=1)
            t_batch = timestep.expand(inputs.shape[0])
            time_bias = self.time_proj(self.time_embed(t_batch)).unsqueeze(-1).unsqueeze(-1)
            net_in = net_in + time_bias
            noise = self.noise_net(net_in)
            sample = self.scheduler.step(noise, timestep, sample).prev_sample
        return sample.clamp(0, 1)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Train the network to predict added Gaussian noise."""
        inputs = batch["inputs"]
        target = batch["target"]
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (target.shape[0],),
            device=target.device,
        ).long()
        noisy_target = self.scheduler.add_noise(target, noise, timesteps)
        net_input = torch.cat([inputs, noisy_target], dim=1)
        time_bias = self.time_proj(self.time_embed(timesteps)).unsqueeze(-1).unsqueeze(-1)
        net_input = net_input + time_bias
        prediction = self.noise_net(net_input)
        loss = F.mse_loss(prediction, noise)
        self.log("train/noise_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validate the denoiser on a held-out noise-prediction batch."""
        x, y = batch["inputs"], batch["target"]
        noise = torch.randn_like(y)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (y.shape[0],), device=y.device,
        ).long()
        noisy_target = self.scheduler.add_noise(y, noise, timesteps)
        net_input = torch.cat([x, noisy_target], dim=1)
        time_bias = self.time_proj(self.time_embed(timesteps)).unsqueeze(-1).unsqueeze(-1)
        net_input = net_input + time_bias
        prediction = self.noise_net(net_input)
        loss = F.mse_loss(prediction, noise)
        self.log("val/noise_loss", loss, prog_bar=True, sync_dist=True)
        if batch_idx % 50 == 0:
            pred = self(x).clamp(0, 1)
            self.log("val/psnr", peak_signal_noise_ratio(pred, y, data_range=1.0))
            self.log("val/ssim", structural_similarity_index_measure(pred, y, data_range=1.0))

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Evaluate reconstruction quality on test data via full denoising."""
        x, y = batch["input"], batch["target"]
        pred = self(x).clamp(0, 1)
        loss, _ = self.reconstruction_loss(pred, y)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/psnr", peak_signal_noise_ratio(pred, y, data_range=1.0))
        self.log("test/ssim", structural_similarity_index_measure(pred, y, data_range=1.0))
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create AdamW optimizer for the diffusion denoiser."""
        optimizer = torch.optim.AdamW(
            self.noise_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.get('max_epochs', 150))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
