# Installation Guide

## Python

Use Python 3.12. On machines with GDAL already configured, `pip` is sufficient:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For geospatial workstations, Conda is often more reliable because GDAL, Rasterio, and GeoPandas share native libraries:

```bash
conda env create -f environment.yml
conda activate uncloud-it
```

## Environment

Copy `.env.example` to `.env` and set credentials locally:

```bash
BHOONIDHI_USERNAME=...
BHOONIDHI_PASSWORD=...
SECRET_KEY=...
DATABASE_URL=postgresql+psycopg://uncloud:uncloud@localhost:5432/uncloud
```

Credentials are never hardcoded in source files.

## Services

Start the full stack:

```bash
docker compose up --build
```

Local development:

```bash
uvicorn backend.app.main:app --reload
cd frontend
npm install
npm run dev
```
