# Inference Guide

## Backend Inference

Start the API:

```bash
uvicorn backend.app.main:app --reload
```

Upload a GeoTIFF:

```bash
curl -F "file=@scene.tif" http://localhost:8000/upload
```

Run reconstruction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"job_id\":\"<job-id>\"}"
```

Download:

```bash
curl -L "http://localhost:8000/download/<job-id>?asset=output" -o reconstructed.tif
```

If `backend.model_checkpoint` does not exist, the service produces a clearly labelled classical inpainting fallback for local smoke testing. Production deployments should mount a trained checkpoint at the configured path.

## CLI Evaluation

```bash
uncloud-evaluate --predictions outputs/predictions --split test
uncloud-evaluate --predictions outputs/predictions --split test --deep-metrics
```

Deep metrics enable LPIPS and FID and may download model weights through TorchMetrics.
