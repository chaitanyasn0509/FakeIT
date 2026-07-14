"""Tests for auxiliary STAC catalogue utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from datasets.auxiliary import (
    LocalHistoricalCatalog,
    StacSearchRequest,
    create_auxiliary_client,
    select_closest_item,
)


def test_stac_search_request_payload() -> None:
    """STAC search requests serialize bbox and datetime fields."""
    request = StacSearchRequest(
        collections=["sentinel-2-l2a"],
        datetime="2024-01-01T00:00:00Z/2024-01-31T23:59:59Z",
        bbox=[77.0, 12.8, 77.8, 13.4],
        limit=25,
    )
    payload = request.to_payload()
    assert payload["collections"] == ["sentinel-2-l2a"]
    assert payload["bbox"] == [77.0, 12.8, 77.8, 13.4]
    assert payload["limit"] == 25


def test_select_closest_item_prefers_nearest_datetime() -> None:
    """The closest STAC item is selected relative to the query interval."""
    items = [
        {"id": "early", "properties": {"datetime": "2024-01-01T00:00:00Z"}},
        {"id": "middle", "properties": {"datetime": "2024-01-15T12:00:00Z"}},
        {"id": "late", "properties": {"datetime": "2024-02-01T00:00:00Z"}},
    ]
    selected = select_closest_item(items, "2024-01-15T00:00:00Z/2024-01-16T00:00:00Z")
    assert selected is not None
    assert selected["id"] == "middle"


def test_create_auxiliary_client_planetary_computer() -> None:
    """Factory creates Planetary Computer clients for Sentinel and DEM modalities."""
    config = {
        "auxiliary": {
            "provider": "planetary_computer",
            "collections": {"sentinel2": "sentinel-2-l2a"},
            "asset_preferences": {"sentinel2": ["B04", "B03", "B02"]},
        }
    }
    client = create_auxiliary_client("sentinel2", config)
    assert client.collection == "sentinel-2-l2a"
    assert client.asset_keys == ["B04", "B03", "B02"]


def test_local_historical_catalog_lists_tifs(tmp_path: Path) -> None:
    """Local historical catalog exposes on-disk GeoTIFFs as STAC-like items."""
    scene = tmp_path / "scene001__historical.tif"
    scene.write_bytes(b"")
    catalog = LocalHistoricalCatalog(tmp_path)
    items = catalog.search(
        StacSearchRequest(
            collections=[],
            datetime=f"{datetime.now(timezone.utc).isoformat()}Z",
            limit=5,
        )
    )
    assert len(items) == 1
    assert items[0]["id"] == "scene001__historical"


def test_create_auxiliary_client_unknown_provider() -> None:
    """Unsupported auxiliary providers raise a clear configuration error."""
    config = {"auxiliary": {"provider": "unknown-provider"}}
    with pytest.raises(ValueError, match="Unsupported auxiliary provider"):
        create_auxiliary_client("sentinel1", config)
