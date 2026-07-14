"""Generate evaluation reports from prediction GeoTIFFs and manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import rasterio
import torch
import typer

from common.config import load_config
from common.logging import configure_logging
from evaluation.metrics import DeepMetricAccumulator, compute_metrics, summarize_metric_table

app = typer.Typer(help="Evaluate reconstructed cloud-free GeoTIFF outputs.")


def read_raster(path: str | Path) -> object:
    """Read a GeoTIFF as a float32 band-first array."""
    with rasterio.open(path) as src:
        return src.read().astype("float32")


@app.command()
def evaluate(
    predictions_dir: Path = typer.Option(Path("outputs/predictions"), "--predictions"),
    manifest_path: Path | None = typer.Option(None, "--manifest"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    split: str = typer.Option("test", "--split"),
    deep_metrics: bool = typer.Option(False, "--deep-metrics/--no-deep-metrics"),
) -> None:
    """Compute metrics for predictions matching manifest scene IDs."""
    configure_logging()
    config = load_config(config_path)
    manifest_file = manifest_path or Path(config["data"]["manifest_path"])
    output_dir = Path(config["evaluation"]["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_file)
    rows: list[dict[str, float | str]] = []
    deep = DeepMetricAccumulator("cuda" if torch.cuda.is_available() else "cpu") if deep_metrics else None
    for _, row in manifest[manifest["split"] == split].iterrows():
        scene_id = str(row["scene_id"])
        prediction_path = predictions_dir / f"{scene_id}.tif"
        if not prediction_path.exists():
            continue
        prediction = read_raster(prediction_path)
        target = read_raster(row["cloud_free_liss4"])
        metrics = compute_metrics(prediction, target)
        if deep is not None:
            deep.update(prediction, target)
        rows.append({"scene_id": scene_id, **metrics})
    table = pd.DataFrame(rows)
    table_path = output_dir / f"{split}_metrics.csv"
    summary_path = output_dir / f"{split}_summary.json"
    summary = summarize_metric_table(rows)
    if deep is not None:
        summary.update(deep.compute())
    table.to_csv(table_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {table_path} and {summary_path}")


def main() -> None:
    """Run the Typer CLI entry point."""
    app()


if __name__ == "__main__":
    main()
