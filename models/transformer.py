"""Transformer-based reconstruction model for multi-modal imagery."""

from __future__ import annotations

import torch
from torch import nn

from models.base import BaseReconstructionModule


class PatchTransformerReconstructor(BaseReconstructionModule):
    """Vision-transformer style reconstructor with convolutional patch embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        learning_rate: float,
        weight_decay: float,
        loss_weights: dict[str, float],
        patch_size: int = 8,
        depth: int = 6,
        heads: int = 8,
    ) -> None:
        """Create the transformer encoder and convolutional decoder."""
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_weights=loss_weights,
        )
        self.save_hyperparameters()
        self.patch_size = patch_size
        embed_dim = base_channels * 4
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, base_channels * 2, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct cloud-free bands from transformer-encoded patch tokens."""
        features = self.embed(inputs)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens)
        encoded_map = encoded.transpose(1, 2).reshape(batch, channels, height, width)
        return torch.sigmoid(self.decoder(encoded_map))
