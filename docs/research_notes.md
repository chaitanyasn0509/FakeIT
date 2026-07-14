# Research Notes

## Research Hypothesis

Combining optical context, SAR backscatter, terrain priors, and historical cloud-free observations improves reconstruction fidelity in cloud-covered LISS-IV imagery compared with optical-only baselines.

## Baselines

1. U-Net with only cloudy LISS-IV.
2. U-Net with all modalities.
3. cGAN with all modalities.
4. Diffusion model with all modalities.
5. Transformer with all modalities.

## Evaluation Protocol

Report spatial metrics, perceptual metrics, and spectral metrics:

- PSNR, SSIM, MAE, RMSE
- LPIPS, FID
- SAM, ERGAS, UIQI

Use geographically separated validation and test areas whenever possible to avoid spatial leakage.

## Remote-Sensing Risks

- Temporal mismatch between target and auxiliary acquisitions can hallucinate changed land cover.
- SAR speckle and DEM artifacts can bias reconstructions near water, steep terrain, and urban edges.
- Cloud-free targets are not always true ground truth because acquisition dates differ.
- Generated imagery must be marked as reconstructed data in downstream GIS workflows.

## Reproducibility

Track:

- Bhoonidhi collection ID and item ID
- download timestamp
- preprocessing config hash
- model version and checkpoint
- train/validation/test scene IDs
- metric script version
