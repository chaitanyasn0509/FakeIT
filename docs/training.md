# Training Guide

## Model Selection

Set `model.name` in `config/default.yaml` to one of:

- `unet`
- `cgan`
- `diffusion`
- `transformer`

Or override on the CLI:

```bash
uncloud-train --config config/default.yaml --model transformer
```

## Lightning Features

The trainer uses:

- automatic GPU detection
- multi-GPU device selection through Lightning
- mixed precision
- gradient accumulation
- checkpointing
- resume training
- TensorBoard
- optional Weights & Biases when `WANDB_API_KEY` is set
- early stopping
- cosine learning-rate scheduling

Resume:

```bash
uncloud-train --config config/default.yaml --resume checkpoints/last.ckpt
```

## Loss Weights

Weighted loss combinations are controlled in YAML:

```yaml
loss:
  l1: 1.0
  ssim: 0.3
  perceptual: 0.05
  edge: 0.1
  spectral: 0.2
  adversarial: 0.01
```

Set unused terms to `0`.
