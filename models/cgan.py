"""Conditional GAN for cloud-free LISS-IV reconstruction."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from models.components import PatchDiscriminator
from models.losses import AdversarialLoss, CompositeReconstructionLoss
from models.unet import UNetReconstructor


class ConditionalGAN(pl.LightningModule):
    """Pix2Pix-style cGAN with multi-modal conditioning and reconstruction losses."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        learning_rate: float,
        weight_decay: float,
        loss_weights: dict[str, float],
    ) -> None:
        """Create generator, discriminator, and manual optimization objectives."""
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.generator = UNetReconstructor(
            in_channels,
            out_channels,
            base_channels,
            learning_rate,
            weight_decay,
            {name: value for name, value in loss_weights.items() if name != "adversarial"},
        )
        self.discriminator = PatchDiscriminator(in_channels + out_channels, base_channels)
        self.reconstruction_loss = CompositeReconstructionLoss(
            {name: value for name, value in loss_weights.items() if name != "adversarial"}
        )
        self.adversarial_loss = AdversarialLoss()
        self.adversarial_weight = float(loss_weights.get("adversarial", 0.01))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate cloud-free imagery from multi-modal conditioning bands."""
        return self.generator(inputs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Train generator and discriminator with manual optimization."""
        generator_optimizer, discriminator_optimizer = self.optimizers()
        inputs = batch["inputs"]
        target = batch["target"]

        prediction = self(inputs)
        fake_logits = self.discriminator(inputs, prediction)
        reconstruction, terms = self.reconstruction_loss(prediction, target)
        generator_loss = reconstruction + self.adversarial_weight * self.adversarial_loss(fake_logits, True)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()

        real_logits = self.discriminator(inputs, target)
        fake_logits = self.discriminator(inputs, prediction.detach())
        discriminator_loss = 0.5 * (
            self.adversarial_loss(real_logits, True) + self.adversarial_loss(fake_logits, False)
        )
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        self.log("train/generator_loss", generator_loss, prog_bar=True, sync_dist=True)
        self.log("train/discriminator_loss", discriminator_loss, sync_dist=True)
        for name, value in terms.items():
            self.log(f"train/loss_{name}", value, sync_dist=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validate the generator using reconstruction loss only."""
        prediction = self(batch["inputs"])
        loss, _terms = self.reconstruction_loss(prediction, batch["target"])
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Evaluate generator quality on test data."""
        x, y = batch["input"], batch["target"]
        pred = self.generator(x)
        loss, _ = self.reconstruction_loss(pred, y)
        pred_clamped = pred.clamp(0, 1)
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/psnr", peak_signal_noise_ratio(pred_clamped, y, data_range=1.0))
        self.log("test/ssim", structural_similarity_index_measure(pred_clamped, y, data_range=1.0))
        return loss

    def configure_optimizers(self):
        """Return paired (G, D) optimizers with cosine-annealing LR schedulers."""
        lr = self.hparams.get("learning_rate", 2e-4)
        wd = self.hparams.get("weight_decay", 1e-5)
        t_max = self.hparams.get("max_epochs", 150)
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, weight_decay=wd, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, weight_decay=wd, betas=(0.5, 0.999))
        sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=t_max)
        sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=t_max)
        return (
            [opt_g, opt_d],
            [{"scheduler": sch_g, "monitor": "val/loss"}, {"scheduler": sch_d, "monitor": "val/loss"}],
        )
