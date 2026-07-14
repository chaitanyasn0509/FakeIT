"""Raster preprocessing primitives for LISS-IV and auxiliary imagery."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

LOGGER = logging.getLogger(__name__)


def reproject_raster(
    input_path: str | Path,
    output_path: str | Path,
    target_crs: str,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster to the requested coordinate reference system."""
    src_path = Path(input_path)
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform, width=width, height=height)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )
    return dst_path


def align_to_reference(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Co-register a raster to the grid, CRS, and dimensions of a reference raster."""
    src_path = Path(input_path)
    ref_path = Path(reference_path)
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        profile = src.profile.copy()
        profile.update(
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=rasterio.band(dst, band_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resampling,
                )
    return dst_path


def clip_to_aoi(input_path: str | Path, aoi_path: str | Path, output_path: str | Path) -> Path:
    """Clip a raster to a vector AOI and preserve georeferencing metadata."""
    src_path = Path(input_path)
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        aoi = gpd.read_file(aoi_path).to_crs(src.crs)
        clipped, transform = mask(src, aoi.geometry, crop=True)
        profile = src.profile.copy()
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(clipped)
    return dst_path


def normalize_raster(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    nodata: float | None = None,
) -> Path:
    """Normalize raster bands to [0, 1] using robust per-band percentiles."""
    src_path = Path(input_path)
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        data = src.read().astype("float32")
        mask_valid = np.ones(data.shape[1:], dtype=bool)
        effective_nodata = nodata if nodata is not None else src.nodata
        if effective_nodata is not None:
            mask_valid = np.all(data != effective_nodata, axis=0)
        normalized = normalize_array(
            data,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            valid_mask=mask_valid,
        )
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=0.0)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(normalized.astype("float32"))
    return dst_path


def normalize_array(
    data: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a band-first array with robust percentile scaling."""
    result = np.zeros_like(data, dtype="float32")
    if valid_mask is None:
        valid_mask = np.ones(data.shape[1:], dtype=bool)
    for band_index in range(data.shape[0]):
        band = data[band_index]
        valid_values = band[valid_mask]
        if valid_values.size == 0:
            continue
        low, high = np.percentile(valid_values, [lower_percentile, upper_percentile])
        denominator = max(float(high - low), 1e-6)
        result[band_index] = np.clip((band - low) / denominator, 0.0, 1.0)
    return result


def generate_patches(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: int,
    stride: int,
    nodata: float | None = None,
    max_nodata_fraction: float = 0.2,
) -> list[Path]:
    """Split a raster into overlapping GeoTIFF patches with geospatial windows."""
    src_path = Path(input_path)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with rasterio.open(src_path) as src:
        effective_nodata = nodata if nodata is not None else src.nodata
        for row in range(0, max(1, src.height - patch_size + 1), stride):
            for col in range(0, max(1, src.width - patch_size + 1), stride):
                window = Window(col_off=col, row_off=row, width=patch_size, height=patch_size)
                patch = src.read(window=window, boundless=False)
                if patch.shape[1:] != (patch_size, patch_size):
                    continue
                if _nodata_fraction(patch, effective_nodata) > max_nodata_fraction:
                    continue
                profile = src.profile.copy()
                profile.update(
                    width=patch_size,
                    height=patch_size,
                    transform=src.window_transform(window),
                )
                patch_path = dst_dir / f"{src_path.stem}__r{row:06d}_c{col:06d}.tif"
                with rasterio.open(patch_path, "w", **profile) as dst:
                    dst.write(patch)
                written.append(patch_path)
    LOGGER.info("Generated %d patches from %s", len(written), src_path)
    return written


def stack_modalities(paths: list[str | Path], output_path: str | Path) -> Path:
    """Stack co-registered rasters into one multi-band GeoTIFF."""
    dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    profile: dict | None = None
    for path in paths:
        with rasterio.open(path) as src:
            arrays.append(src.read())
            if profile is None:
                profile = src.profile.copy()
    if profile is None:
        raise ValueError("At least one input raster is required for stacking.")
    stacked = np.concatenate(arrays, axis=0)
    profile.update(count=stacked.shape[0])
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(stacked)
    return dst_path


def _nodata_fraction(patch: np.ndarray, nodata: float | None) -> float:
    """Compute the fraction of patch pixels that should be considered nodata."""
    if nodata is None:
        return 0.0
    nodata_pixels = np.all(patch == nodata, axis=0)
    return float(nodata_pixels.mean())
