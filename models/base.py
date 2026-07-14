"""Common PyTorch Lightning interface for reconstruction models."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

from models.losses import CompositeReconstructionLoss


class BaseReconstructionModule(pl.LightningModule):
    """Base class for supervised cloud-free image reconstruction."""

    def __init__(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        loss_weights: dict[str, float],
    ) -> None:
        """Initialize optimizer settings and composite reconstruction loss."""
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reconstruction_loss = CompositeReconstructionLoss(loss_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct cloud-free LISS-IV bands from a multi-modal tensor."""
        raise NotImplementedError

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one supervised optimization step."""
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Evaluate a validation batch and log image-quality metrics."""
        self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Evaluate a test batch and log reconstruction metrics."""
        self._shared_step(batch, stage="test")

    def _shared_step(self, batch: dict[str, torch.Tensor], *, stage: str) -> torch.Tensor:
        """Compute prediction, losses, and common quality metrics."""
        inputs = batch["inputs"]
        target = batch["target"]
        prediction = self(inputs)
        loss, terms = self.reconstruction_loss(prediction, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        for name, value in terms.items():
            self.log(f"{stage}/loss_{name}", value, sync_dist=True)
        self.log(
            f"{stage}/psnr",
            peak_signal_noise_ratio(prediction.clamp(0, 1), target.clamp(0, 1), data_range=1.0),
            sync_dist=True,
        )
        self.log(
            f"{stage}/ssim",
            structural_similarity_index_measure(
                prediction.clamp(0, 1),
                target.clamp(0, 1),
                data_range=1.0,
            ),
            prog_bar=stage == "val",
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Create AdamW optimizer with cosine annealing schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.get('max_epochs', 150))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
