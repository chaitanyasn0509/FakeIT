# Architecture Diagram

```mermaid
flowchart LR
  User["Researcher / Analyst"] --> Frontend["Next.js Frontend"]
  Frontend --> API["FastAPI Backend"]
  API --> Storage["Local or S3-Compatible Storage"]
  API --> DB["PostgreSQL"]
  API --> Inference["Inference Service"]
  Inference --> Checkpoint["Lightning Checkpoint"]
  Inference --> Masks["Cloud Mask Generator"]
  Inference --> GeoTIFF["Reconstructed GeoTIFF"]

  Bhoonidhi["Official Bhoonidhi API"] --> Client["Bhoonidhi Client"]
  Client --> Raw["Raw Data Lake"]
  Raw --> Preprocess["Preprocessing Pipeline"]
  Auxiliary["Sentinel-1 / Sentinel-2 / DEM / Historical LISS-IV"] --> Preprocess
  Preprocess --> Manifest["Multi-Modal Manifest"]
  Manifest --> Training["Lightning Training"]
  Training --> Models["U-Net / cGAN / Diffusion / Transformer"]
  Models --> Checkpoint
  Models --> Evaluation["Metrics and Reports"]
```

## Data Contract

Training batches use:

- `inputs`: stacked cloudy LISS-IV, Sentinel-1, Sentinel-2, DEM, historical LISS-IV
- `target`: cloud-free LISS-IV
- `scene_id`: reproducibility identifier

The backend accepts a GeoTIFF upload and runs mask generation plus model inference. For full multi-modal production inference, prepare a stacked input GeoTIFF with the same channel order used during training.
