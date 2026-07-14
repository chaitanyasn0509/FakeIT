# Dataset Preparation Guide

## Bhoonidhi LISS-IV Acquisition

The official Bhoonidhi API base URL used by the code is `https://bhoonidhi-api.nrsc.gov.in`.

Prepare a STAC query JSON with collection, time range, AOI, and `limit <= 500`. The downloader adds an `Online = Y` CQL2 filter when no filter is provided.

```json
{
  "collections": ["RESOURCESAT-2_LISS4_MX_L2"],
  "datetime": "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z",
  "intersects": {
    "type": "Polygon",
    "coordinates": [[[77.0, 12.8], [77.8, 12.8], [77.8, 13.4], [77.0, 13.4], [77.0, 12.8]]]
  },
  "limit": 100
}
```

Run:

```bash
python -m scripts.download_bhoonidhi --query query.json --output-dir data/raw/bhoonidhi
```

## Auxiliary Data

Sentinel-1 SAR, Sentinel-2, Copernicus DEM, and historical LISS-IV are fetched through the
[Planetary Computer STAC API](https://planetarycomputer.microsoft.com/) by default. Configure an
optional subscription key through `PLANETARY_COMPUTER_SUBSCRIPTION_KEY` for higher throughput.

Search one modality:

```bash
python -m scripts.download_auxiliary search \
  --modality sentinel2 \
  --bbox 77.0,12.8,77.8,13.4 \
  --datetime 2024-06-01T00:00:00Z/2024-06-30T23:59:59Z
```

Download and co-register to a processed LISS-IV reference grid:

```bash
python -m scripts.download_auxiliary download \
  --modality sentinel2 \
  --datetime 2024-06-15T00:00:00Z/2024-06-16T00:00:00Z \
  --align-to data/processed/cloudy_liss4/normalized/scene001.tif \
  --output-dir data/processed/sentinel2 \
  --scene-id scene001
```

Fetch all configured auxiliary modalities for one scene:

```bash
python -m scripts.download_auxiliary fetch-for-scene \
  --reference-scene data/processed/cloudy_liss4/normalized/scene001.tif
```

Or run preprocessing and auxiliary pairing together:

```bash
python -m scripts.run_pipeline prepare-scene \
  --scene data/raw/bhoonidhi/scene001.tif \
  --fetch-auxiliary
```

Store historical LISS-IV under `data/raw/historical_liss4/` for the local catalogue adapter.
Processed auxiliary products are written to:

```text
data/processed/sentinel1/
data/processed/sentinel2/
data/processed/dem/
data/processed/historical_liss4/
```

Use `preprocessing.pipeline.PreprocessingPipeline.align_auxiliary` to co-register manually when needed.

## Patch Manifest

Expected processed groups:

```text
data/processed/cloudy_liss4/
data/processed/cloud_free_liss4/
data/processed/sentinel1/
data/processed/sentinel2/
data/processed/dem/
data/processed/historical_liss4/
```

Files are paired by the scene ID before the first `__` in the filename. Build splits:

```bash
uncloud-build-dataset --config config/default.yaml
```
