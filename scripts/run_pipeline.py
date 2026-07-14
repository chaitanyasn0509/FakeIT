"""End-to-end pipeline orchestration for UNCLOUD IT data preparation."""

from __future__ import annotations

from pathlib import Path

import typer

from common.config import load_config
from common.logging import configure_logging
from datasets.auxiliary import fetch_auxiliary_for_reference
from datasets.builder import ManifestBuilder
from preprocessing.pipeline import PreprocessingPipeline

app = typer.Typer(help="Run download, preprocessing, auxiliary pairing, and manifest generation.")


@app.command()
def prepare_scene(
    scene_path: Path = typer.Option(..., "--scene", help="Raw or downloaded LISS-IV GeoTIFF"),
    cloudy: bool = typer.Option(True, "--cloudy/--cloud-free", help="Whether the scene is cloudy input"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    fetch_auxiliary: bool = typer.Option(True, "--fetch-auxiliary/--skip-auxiliary"),
    build_manifest: bool = typer.Option(False, "--build-manifest"),
) -> None:
    """Preprocess one LISS-IV scene and optionally fetch aligned auxiliary products."""
    configure_logging()
    config = load_config(config_path)
    pipeline = PreprocessingPipeline(config_path)
    group = "cloudy_liss4" if cloudy else "cloud_free_liss4"
    result = pipeline.process_liss4_scene(scene_path, group)
    typer.echo(f"Preprocessed {group}: {result.normalized_path}")
    if fetch_auxiliary and cloudy:
        scene_id = scene_path.stem.split("__")[0]
        processed_root = Path(config["data"]["processed_dir"])
        for modality in config.get("auxiliary", {}).get(
            "modalities",
            ["sentinel1", "sentinel2", "dem", "historical_liss4"],
        ):
            aligned = fetch_auxiliary_for_reference(
                modality=str(modality),
                reference_path=result.normalized_path,
                output_dir=processed_root / modality,
                config=config,
                scene_id=scene_id,
            )
            typer.echo(f"Aligned {modality}: {aligned}")
    if build_manifest:
        manifest_path = ManifestBuilder(
            root=config["data"]["processed_dir"],
            splits=config["dataset"]["splits"],
            seed=int(config["project"].get("seed", 42)),
        ).write(config["data"]["manifest_path"])
        typer.echo(f"Manifest written to {manifest_path}")


def main() -> None:
    """Run the Typer CLI entry point."""
    app()


if __name__ == "__main__":
    main()
