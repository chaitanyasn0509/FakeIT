"""Auxiliary remote-sensing catalogue utilities for multi-modal pairing."""

from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

DEFAULT_COLLECTIONS = {
    "sentinel1": "sentinel-1-grd",
    "sentinel2": "sentinel-2-l2a",
    "dem": "cop-dem-glo-30",
}

DEFAULT_ASSET_KEYS = {
    "sentinel1": ["vh", "vv"],
    "sentinel2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "dem": ["data"],
}


@dataclass(slots=True)
class StacSearchRequest:
    """A generic STAC query used for Sentinel, DEM, and historical searches."""

    collections: list[str]
    datetime: str
    bbox: list[float] | None = None
    intersects: dict[str, Any] | None = None
    limit: int = 100

    def to_payload(self) -> dict[str, Any]:
        """Convert the request to a STAC-compliant JSON search payload."""
        payload: dict[str, Any] = {
            "collections": self.collections,
            "datetime": self.datetime,
            "limit": self.limit,
        }
        if self.bbox:
            payload["bbox"] = self.bbox
        if self.intersects:
            payload["intersects"] = self.intersects
        return payload


class AuxiliaryCatalogClient(ABC):
    """Abstract base for external STAC catalog clients."""

    @abstractmethod
    def search(self, request: StacSearchRequest) -> list[dict[str, Any]]:
        """Search an auxiliary catalogue and return STAC-like item dictionaries."""

    @abstractmethod
    def download(self, item: dict[str, Any], output_dir: str | Path) -> Path:
        """Download or materialize an auxiliary item into the project data lake."""


def sign_planetary_computer_href(href: str, subscription_key: str = "") -> str:
    """Sign a Planetary Computer asset URL for time-limited authenticated access."""
    try:
        import planetary_computer as pc

        if subscription_key:
            os.environ.setdefault("PLANETARY_COMPUTER_SUBSCRIPTION_KEY", subscription_key)
        return pc.sign(href)
    except ImportError:
        headers: dict[str, str] = {}
        if subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = subscription_key
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(
                "https://planetarycomputer.microsoft.com/api/sas/token",
                params={"href": href},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("href", href))


def bbox_from_raster(path: str | Path) -> list[float]:
    """Extract a WGS84 bounding box from a GeoTIFF path."""
    import rasterio

    with rasterio.open(path) as src:
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]


def datetime_from_raster(path: str | Path, *, window_days: int = 7) -> str:
    """Build a STAC datetime interval centered on raster acquisition metadata."""
    import rasterio

    with rasterio.open(path) as src:
        tags = src.tags()
        candidate = tags.get("TIFFTAG_DATETIME") or tags.get("ACQUISITION_DATE") or tags.get("creation_time")
    if candidate:
        try:
            center = datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
        except ValueError:
            center = datetime.utcnow()
    else:
        center = datetime.utcnow()
    start = center.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start.fromordinal(start.toordinal() + window_days)
    return f"{start.isoformat()}Z/{end.isoformat()}Z"


def select_closest_item(items: list[dict[str, Any]], target_datetime: str | None = None) -> dict[str, Any] | None:
    """Pick the STAC item whose acquisition time is closest to a target interval."""
    if not items:
        return None
    if not target_datetime:
        return items[0]
    target = _parse_interval_center(target_datetime)
    if target is None:
        return items[0]

    def item_time(item: dict[str, Any]) -> datetime:
        raw = str(item.get("properties", {}).get("datetime", ""))
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=target.tzinfo)

    return min(items, key=lambda item: abs((item_time(item) - target).total_seconds()))


def _parse_interval_center(interval: str) -> datetime | None:
    """Return the midpoint of a STAC datetime interval string."""
    if "/" not in interval:
        try:
            return datetime.fromisoformat(interval.replace("Z", "+00:00"))
        except ValueError:
            return None
    start_raw, end_raw = interval.split("/", 1)
    try:
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
        return start + (end - start) / 2
    except ValueError:
        return None


class PlanetaryComputerCatalog(AuxiliaryCatalogClient):
    """STAC client for Sentinel-1, Sentinel-2, and Copernicus DEM on Planetary Computer."""

    def __init__(
        self,
        *,
        collection: str,
        asset_keys: list[str],
        stac_url: str = PLANETARY_COMPUTER_STAC_URL,
        subscription_key: str = "",
        timeout_seconds: float = 120.0,
    ) -> None:
        """Create a Planetary Computer catalogue adapter for one modality."""
        self.collection = collection
        self.asset_keys = asset_keys
        self.stac_url = stac_url
        self.subscription_key = subscription_key
        self._client = httpx.Client(timeout=timeout_seconds, follow_redirects=True)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "PlanetaryComputerCatalog":
        """Enter a context-managed catalogue session."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Exit a context-managed catalogue session."""
        self.close()

    def search(self, request: StacSearchRequest) -> list[dict[str, Any]]:
        """Search Planetary Computer STAC and return matching features."""
        payload = request.to_payload()
        payload["collections"] = [self.collection]
        headers: dict[str, str] = {}
        if self.subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = self.subscription_key
        response = self._client.post(self.stac_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("features", [])

    def download(self, item: dict[str, Any], output_dir: str | Path) -> Path:
        """Download selected assets from a STAC item and stack them into one GeoTIFF."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        item_id = str(item.get("id", "item"))
        destination = output_path / f"{item_id}.tif"
        assets = item.get("assets", {})
        available_keys = [key for key in self.asset_keys if key in assets]
        if not available_keys:
            available_keys = _fallback_asset_keys(assets)
        if not available_keys:
            raise ValueError(f"No downloadable assets found for STAC item {item_id}")
        with tempfile.TemporaryDirectory(prefix="uncloud-aux-") as temp_dir:
            band_paths = [
                self._download_asset(assets[key], Path(temp_dir) / f"{key}.tif", label=key)
                for key in available_keys
            ]
            _stack_rasters(band_paths, destination)
        return destination

    def download_and_align(
        self,
        item: dict[str, Any],
        output_dir: str | Path,
        reference_path: str | Path,
        *,
        scene_id: str | None = None,
    ) -> Path:
        """Download a STAC item, stack assets, and co-register to a LISS-IV reference grid."""
        from preprocessing.geospatial import align_to_reference

        raw_path = self.download(item, output_dir)
        stem = scene_id or Path(reference_path).stem.split("__")[0]
        aligned_path = Path(output_dir) / f"{stem}.tif"
        align_to_reference(raw_path, reference_path, aligned_path)
        raw_path.unlink(missing_ok=True)
        return aligned_path

    def _download_asset(self, asset: dict[str, Any], destination: Path, *, label: str) -> Path:
        """Stream one signed STAC asset to disk."""
        href = str(asset["href"])
        signed_href = sign_planetary_computer_href(href, self.subscription_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", signed_href) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", "0") or 0)
            with destination.open("wb") as stream:
                with tqdm(total=total, unit="B", unit_scale=True, desc=label) as progress:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        progress.update(len(chunk))
        return destination


class LocalHistoricalCatalog(AuxiliaryCatalogClient):
    """Catalogue adapter for historical LISS-IV files already present on disk."""

    def __init__(self, root: str | Path) -> None:
        """Create a local catalogue rooted at a directory of GeoTIFF files."""
        self.root = Path(root)

    def search(self, request: StacSearchRequest) -> list[dict[str, Any]]:
        """Return local files as STAC-like items for downstream pairing."""
        return [
            {"id": path.stem, "assets": {"image": {"href": str(path)}}}
            for path in self.root.rglob("*.tif")
        ][: request.limit]

    def download(self, item: dict[str, Any], output_dir: str | Path) -> Path:
        """Return the local path because local historical files are already available."""
        return Path(item["assets"]["image"]["href"])


def create_auxiliary_client(modality: str, config: dict[str, Any]) -> AuxiliaryCatalogClient:
    """Instantiate an auxiliary catalogue client from project configuration."""
    auxiliary_config = config.get("auxiliary", {})
    provider = str(auxiliary_config.get("provider", "planetary_computer")).lower()
    if modality == "historical_liss4":
        root = auxiliary_config.get("historical_root", "data/raw/historical_liss4")
        return LocalHistoricalCatalog(root)
    if provider != "planetary_computer":
        raise ValueError(f"Unsupported auxiliary provider: {provider}")
    collections = auxiliary_config.get("collections", DEFAULT_COLLECTIONS)
    asset_preferences = auxiliary_config.get("asset_preferences", DEFAULT_ASSET_KEYS)
    collection = str(collections.get(modality, DEFAULT_COLLECTIONS[modality]))
    asset_keys = list(asset_preferences.get(modality, DEFAULT_ASSET_KEYS[modality]))
    return PlanetaryComputerCatalog(
        collection=collection,
        asset_keys=asset_keys,
        stac_url=str(auxiliary_config.get("stac_url", PLANETARY_COMPUTER_STAC_URL)),
        subscription_key=str(auxiliary_config.get("subscription_key", "")),
        timeout_seconds=float(auxiliary_config.get("timeout_seconds", 120)),
    )


def fetch_auxiliary_for_reference(
    *,
    modality: str,
    reference_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    bbox: list[float] | None = None,
    datetime_interval: str | None = None,
    scene_id: str | None = None,
) -> Path:
    """Search, download, and align one auxiliary modality for a LISS-IV reference scene."""
    reference = Path(reference_path)
    client = create_auxiliary_client(modality, config)
    search_request = StacSearchRequest(
        collections=[],
        datetime=datetime_interval or datetime_from_raster(reference),
        bbox=bbox or bbox_from_raster(reference),
        limit=int(config.get("auxiliary", {}).get("search_limit", 50)),
    )
    if isinstance(client, PlanetaryComputerCatalog):
        items = client.search(search_request)
        item = select_closest_item(items, search_request.datetime)
        if item is None:
            raise RuntimeError(f"No {modality} STAC items found for {reference}")
        return client.download_and_align(
            item,
            output_dir,
            reference,
            scene_id=scene_id or reference.stem.split("__")[0],
        )
    if isinstance(client, LocalHistoricalCatalog):
        from preprocessing.geospatial import align_to_reference

        items = client.search(search_request)
        item = select_closest_item(items, search_request.datetime)
        if item is None:
            raise RuntimeError(f"No local historical LISS-IV files found under {client.root}")
        source = client.download(item, output_dir)
        stem = scene_id or reference.stem.split("__")[0]
        aligned = Path(output_dir) / f"{stem}.tif"
        align_to_reference(source, reference, aligned)
        return aligned
    raise TypeError(f"Unsupported auxiliary client type: {type(client)}")


def _fallback_asset_keys(assets: dict[str, Any]) -> list[str]:
    """Select raster-like assets when configured keys are unavailable."""
    preferred = ("data", "visual", "image", "elevation")
    for key in preferred:
        if key in assets:
            return [key]
    for key, asset in assets.items():
        media_type = str(asset.get("type", ""))
        if media_type.startswith("image/") or key.lower().endswith(("tif", "tiff", "cog")):
            return [key]
    return [next(iter(assets))] if assets else []


def _stack_rasters(paths: list[Path], destination: Path) -> Path:
    """Stack single-band rasters into one multi-band GeoTIFF on a common grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if len(paths) == 1:
        paths[0].replace(destination)
        return destination
    reference_path = paths[0]
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        stacked = [ref.read(1).astype("float32")]
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = ref.shape
    for path in paths[1:]:
        with rasterio.open(path) as src:
            aligned = np.zeros(ref_shape, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            stacked.append(aligned)
    profile.update(count=len(stacked), dtype="float32")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        for index, band in enumerate(stacked, start=1):
            dst.write(band, index)
    return destination
