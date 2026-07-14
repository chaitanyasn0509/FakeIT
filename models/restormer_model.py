from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

from models.restormer.restormer_arch import Restormer


class RestormerReconstructor(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        loss_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = Restormer(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=24,
            num_blocks=[2, 2, 2, 4],
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
        )

        self.lr = learning_rate
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights or {"l1": 1.0}

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x = batch["inputs"]
        y = batch["target"]

        pred = self(x)

        loss = F.l1_loss(pred, y)

        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["inputs"].size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):

        loss, pred, target = self._step(batch)

        ssim = self.ssim(pred, target)

        self.log("val/loss", loss, on_epoch=True)
        self.log("val/ssim", ssim, on_epoch=True)

        print("LOSS =", loss.item())
        print("SSIM =", ssim.item())
        print(self.trainer.callback_metrics)
        return loss
    




    def test_step(self, batch, batch_idx):

        loss, pred, target = self._step(batch)

        ssim = self.ssim(pred, target)
        psnr = self.psnr(pred, target)
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))

        self.log("test/loss", loss)
        self.log("test/ssim", ssim)
        self.log("test/psnr", psnr)
        self.log("test/rmse", rmse)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }