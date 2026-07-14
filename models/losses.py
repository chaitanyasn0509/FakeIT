"""Loss functions for remote-sensing reconstruction quality."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision import models as tv_models


class SSIMLoss(nn.Module):
    """Differentiable structural similarity loss."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return one minus SSIM for normalized tensors."""
        value = structural_similarity_index_measure(prediction.clamp(0, 1), target.clamp(0, 1))
        return 1.0 - value


class EdgeLoss(nn.Module):
    """Sobel edge consistency loss for preserving roads, rivers, and field boundaries."""

    def __init__(self) -> None:
        """Create fixed Sobel kernels."""
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare Sobel gradient magnitudes between prediction and target."""
        pred_edges = self._edges(prediction)
        target_edges = self._edges(target)
        return F.l1_loss(pred_edges, target_edges)

    def _edges(self, image: torch.Tensor) -> torch.Tensor:
        """Compute per-channel Sobel gradient magnitudes."""
        channels = image.shape[1]
        kernel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


class SpectralConsistencyLoss(nn.Module):
    """Spectral-angle loss for preserving band relationships."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean spectral angle mapper loss in radians."""
        pred = prediction.flatten(start_dim=2)
        truth = target.flatten(start_dim=2)
        numerator = (pred * truth).sum(dim=1)
        denominator = pred.norm(dim=1).clamp_min(1e-6) * truth.norm(dim=1).clamp_min(1e-6)
        cosine = (numerator / denominator).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.acos(cosine).mean()


class PerceptualLoss(nn.Module):
    """VGG feature-space loss for visual realism in RGB-like LISS-IV bands."""

    def __init__(self, pretrained: bool = False) -> None:
        """Create a frozen VGG16 feature extractor."""
        super().__init__()
        weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = tv_models.vgg16(weights=weights).features[:16].eval()
        for parameter in features.parameters():
            parameter.requires_grad = False
        self.features = features

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare VGG features after adapting tensors to three channels."""
        pred = self._to_three_channels(prediction)
        truth = self._to_three_channels(target)
        return F.l1_loss(self.features(pred), self.features(truth))

    def _to_three_channels(self, image: torch.Tensor) -> torch.Tensor:
        """Select or repeat channels so VGG receives three-channel inputs."""
        if image.shape[1] == 3:
            return image
        if image.shape[1] > 3:
            return image[:, :3]
        return image.repeat(1, 3, 1, 1)[:, :3]


class AdversarialLoss(nn.Module):
    """Binary cross-entropy adversarial objective for PatchGAN outputs."""

    def __init__(self) -> None:
        """Create the BCE-with-logits loss."""
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compare logits to real or fake patch labels."""
        labels = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.loss(logits, labels)


class CompositeReconstructionLoss(nn.Module):
    """Weighted reconstruction loss configured from YAML."""

    def __init__(self, weights: dict[str, float]) -> None:
        """Create all requested loss terms and remember their weights."""
        super().__init__()
        self.weights = {name: float(value) for name, value in weights.items() if float(value) > 0}
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.spectral = SpectralConsistencyLoss()
        self.perceptual = PerceptualLoss(pretrained=True)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return weighted total loss and detached term values for logging."""
        losses: dict[str, torch.Tensor] = {}
        if "l1" in self.weights:
            losses["l1"] = self.l1(prediction, target)
        if "ssim" in self.weights:
            losses["ssim"] = self.ssim(prediction, target)
        if "edge" in self.weights:
            losses["edge"] = self.edge(prediction, target)
        if "spectral" in self.weights:
            losses["spectral"] = self.spectral(prediction, target)
        if "perceptual" in self.weights:
            losses["perceptual"] = self.perceptual(prediction, target)
        total = sum(self.weights[name] * value for name, value in losses.items())
        logged = {name: value.detach() for name, value in losses.items()}
        return total, logged
