"""Reusable neural-network components for reconstruction models."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two-layer convolution block with normalization and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a convolutional feature extractor."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform an image tensor into local convolutional features."""
        return self.net(x)


class DownBlock(nn.Module):
    """Downsampling block for encoder stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a max-pool plus convolution stage."""
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and encode feature maps."""
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling block with skip-connection fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a transposed-convolution upsampling stage."""
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x, concatenate the skip map, and refine the result."""
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for conditional adversarial training."""

    def __init__(self, in_channels: int, base_channels: int = 64) -> None:
        """Create a discriminator that scores local image realism."""
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        previous = in_channels
        for index, channel in enumerate(channels):
            stride = 1 if index == len(channels) - 1 else 2
            layers.extend(
                [
                    nn.Conv2d(previous, channel, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(channel),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            previous = channel
        layers.append(nn.Conv2d(previous, 1, kernel_size=4, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, conditioning: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Score a conditioned image as real or generated at patch level."""
        return self.net(torch.cat([conditioning, image], dim=1))
