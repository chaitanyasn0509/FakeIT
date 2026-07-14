# UNCLOUD IT

Generative AI-Based Cloud Removal and Surface Reconstruction for LISS-IV Satellite Imagery using Multi-Modal Remote Sensing Data.

This repository is structured as a research-grade project: official Bhoonidhi API access, geospatial preprocessing, multi-modal dataset pairing, modular reconstruction models, Lightning training, quantitative evaluation, a FastAPI backend, a Next.js frontend, and containerized deployment.

## Core Capabilities

- Bhoonidhi API client with `/auth/token`, refresh-token flow, STAC collection/search, metadata retrieval, online-product filtering, throttling-aware retries, and `/download`.
- Preprocessing for reprojection, AOI clipping, co-registration, normalization, patch generation, and cloud masks through thresholding, FMask, or deep segmentation checkpoints.
- Multi-modal dataset builder for cloudy LISS-IV, cloud-free LISS-IV, Sentinel-1 SAR, Sentinel-2, DEM, and historical LISS-IV.
- Models: U-Net baseline, conditional GAN, conditional diffusion model, and transformer reconstructor.
- Losses: L1, SSIM, perceptual, adversarial, edge, and spectral consistency.
- Metrics: PSNR, SSIM, LPIPS, MAE, RMSE, SAM, ERGAS, UIQI, and FID.
- FastAPI backend and Next.js frontend for upload, cloud detection, reconstruction, metrics, history, and GeoTIFF download.

## Quick Start

```bash
cp .env.example .env
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.download_bhoonidhi --query docs/example_bhoonidhi_query.json
python -m scripts.run_pipeline prepare-scene --scene data/raw/bhoonidhi/example.tif --fetch-auxiliary
python -m scripts.preprocess_scene --scene data/raw/bhoonidhi/example.tif --group cloudy_liss4
uncloud-build-dataset --config config/default.yaml
uncloud-train --config config/default.yaml --model unet
uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Docker:

```bash
docker compose up --build
```

## Repository Layout

```text
backend/          FastAPI app, database models, inference and storage services
frontend/         Next.js geospatial upload and reconstruction UI
models/           U-Net, cGAN, diffusion, transformer, losses, model factory
training/         Lightning DataModule, train CLI, tiled inference
datasets/         Bhoonidhi client, auxiliary catalog interfaces, manifest builder
preprocessing/    CRS, registration, clipping, normalization, masks, patches
evaluation/       Metrics and report generation
visualization/    Qualitative comparison figures
deployment/       Kubernetes starter manifests
docs/             Installation, data, training, inference, architecture, notes
tests/            Unit tests for config, metrics, and API-client behavior
```

## Bhoonidhi Notes

The implementation follows the official Bhoonidhi API specification page: authentication uses `userId`, `password`, and `grant_type=password` at `/auth/token`; refresh uses `grant_type=refresh_token`; STAC search is under `/data/search`; downloads use `/download?id=<id>&collection=<collection>`. Configure credentials only through environment variables.

## Documentation

- [Installation Guide](docs/installation.md)
- [Dataset Preparation Guide](docs/dataset_preparation.md)
- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [API Documentation](docs/api.md)
- [Architecture Diagram](docs/architecture.md)
- [Research Notes](docs/research_notes.md)
