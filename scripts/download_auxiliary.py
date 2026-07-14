"""Download Sentinel-1, Sentinel-2, DEM, or historical LISS-IV auxiliary products."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from common.config import load_config
from common.logging import configure_logging
from datasets.auxiliary import (
    StacSearchRequest,
    bbox_from_raster,
    create_auxiliary_client,
    datetime_from_raster,
    fetch_auxiliary_for_reference,
    select_closest_item,
)

app = typer.Typer(help="Download auxiliary remote-sensing products for multi-modal training.")


@app.command()
def search(
    modality: str = typer.Option(..., "--modality", help="sentinel1, sentinel2, dem, or historical_liss4"),
    bbox: str | None = typer.Option(None, "--bbox", help="minx,miny,maxx,maxy in WGS84"),
    datetime_interval: str = typer.Option(..., "--datetime"),
    output_dir: Path = typer.Option(Path("data/raw/auxiliary"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """Search an auxiliary STAC catalogue and print matching item metadata."""
    configure_logging()
    config = load_config(config_path)
    client = create_auxiliary_client(modality, config)
    parsed_bbox = [float(value) for value in bbox.split(",")] if bbox else None
    request = StacSearchRequest(
        collections=[],
        datetime=datetime_interval,
        bbox=parsed_bbox,
        limit=limit,
    )
    items = client.search(request)
    typer.echo(json.dumps(items, indent=2))


@app.command()
def download(
    modality: str = typer.Option(..., "--modality"),
    bbox: str | None = typer.Option(None, "--bbox"),
    datetime_interval: str = typer.Option(..., "--datetime"),
    output_dir: Path = typer.Option(Path("data/raw/auxiliary"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    align_to: Path | None = typer.Option(None, "--align-to", help="Reference LISS-IV GeoTIFF for co-registration"),
    scene_id: str | None = typer.Option(None, "--scene-id"),
) -> None:
    """Download the closest auxiliary product for a bbox and datetime interval."""
    configure_logging()
    config = load_config(config_path)
    if align_to:
        aligned = fetch_auxiliary_for_reference(
            modality=modality,
            reference_path=align_to,
            output_dir=output_dir,
            config=config,
            bbox=[float(value) for value in bbox.split(",")] if bbox else None,
            datetime_interval=datetime_interval,
            scene_id=scene_id,
        )
        typer.echo(f"Aligned {modality} product saved to {aligned}")
        return

    client = create_auxiliary_client(modality, config)
    parsed_bbox = [float(value) for value in bbox.split(",")] if bbox else None
    request = StacSearchRequest(
        collections=[],
        datetime=datetime_interval,
        bbox=parsed_bbox,
        limit=int(config.get("auxiliary", {}).get("search_limit", 50)),
    )
    items = client.search(request)
    item = select_closest_item(items, datetime_interval)
    if item is None:
        raise typer.Exit(code=1)
    downloaded = client.download(item, output_dir)
    typer.echo(f"Downloaded {modality} product to {downloaded}")


@app.command()
def fetch_for_scene(
    reference_scene: Path = typer.Option(..., "--reference-scene"),
    output_root: Path = typer.Option(Path("data/processed"), "--output-root"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    scene_id: str | None = typer.Option(None, "--scene-id"),
    modalities: str = typer.Option(
        "sentinel1,sentinel2,dem,historical_liss4",
        "--modalities",
        help="Comma-separated auxiliary modalities to fetch",
    ),
) -> None:
    """Download and align all configured auxiliary modalities for one LISS-IV scene."""
    configure_logging()
    config = load_config(config_path)
    stem = scene_id or reference_scene.stem.split("__")[0]
    bbox = bbox_from_raster(reference_scene)
    datetime_interval = datetime_from_raster(reference_scene)
    written: list[str] = []
    for modality in [value.strip() for value in modalities.split(",") if value.strip()]:
        group_dir = output_root / modality
        aligned = fetch_auxiliary_for_reference(
            modality=modality,
            reference_path=reference_scene,
            output_dir=group_dir,
            config=config,
            bbox=bbox,
            datetime_interval=datetime_interval,
            scene_id=stem,
        )
        written.append(f"{modality}: {aligned}")
    typer.echo("\n".join(written))


def main() -> None:
    """Run the Typer CLI entry point."""
    app()


if __name__ == "__main__":
    main()
