"""Run preprocessing for a single LISS-IV scene."""

from __future__ import annotations

from pathlib import Path

import typer

from common.logging import configure_logging
from preprocessing.pipeline import PreprocessingPipeline

app = typer.Typer(help="Preprocess LISS-IV scenes into model-ready patches.")


@app.command()
def preprocess(
    scene_path: Path = typer.Option(..., "--scene"),
    output_group: str = typer.Option("cloudy_liss4", "--group"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
) -> None:
    """Run reprojection, AOI clipping, normalization, masks, and patch generation."""
    configure_logging()
    result = PreprocessingPipeline(config_path).process_liss4_scene(scene_path, output_group)
    typer.echo(f"Normalized: {result.normalized_path}")
    typer.echo(f"Mask: {result.mask_path}")
    typer.echo(f"Patches: {len(result.patch_paths)}")


def main() -> None:
    """Run the Typer CLI entry point."""
    app()


if __name__ == "__main__":
    main()
