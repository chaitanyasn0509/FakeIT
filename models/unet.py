"""U-Net baseline for multi-modal cloud removal."""

from __future__ import annotations

import torch
from torch import nn

from models.base import BaseReconstructionModule
from models.components import ConvBlock, DownBlock, UpBlock


class UNetReconstructor(BaseReconstructionModule):
    """U-Net model that maps stacked multi-modal bands to cloud-free LISS-IV bands."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        learning_rate: float,
        weight_decay: float,
        loss_weights: dict[str, float],
        output_activation: str = "sigmoid",
    ) -> None:
        """Create a four-level U-Net reconstruction network."""
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_weights=loss_weights,
        )
        self.save_hyperparameters()
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.output_activation = output_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate a normalized cloud-free reconstruction."""
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.output_activation == "identity":
            return logits
        return torch.sigmoid(logits)
