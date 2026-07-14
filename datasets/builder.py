from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import typer

from common.config import load_config
from common.logging import configure_logging

app = typer.Typer()


class ManifestBuilder:

    def __init__(self, root: str | Path, splits: dict[str, float], seed: int = 42):
        self.root = Path(root)
        self.splits = splits
        self.seed = seed

    def build(self):

        cloudy_root = self.root / "cloudDNclips" / "10m"
        clear_root = self.root / "clearDNclips" / "10m"

        rows = []

        for cloudy_file in cloudy_root.rglob("*.tif"):

            relative = cloudy_file.relative_to(cloudy_root)

            clear_file = clear_root / relative

            if not clear_file.exists():
                continue

            scene_id = relative.with_suffix("").as_posix().replace("/", "_")

            rows.append(
                {
                    "scene_id": scene_id,
                    "cloudy": str(cloudy_file),
                    "clear": str(clear_file),
                }
            )

        random.Random(self.seed).shuffle(rows)

        n = len(rows)

        train_n = int(n * self.splits["train"])
        val_n = int(n * self.splits["val"])

        for i, row in enumerate(rows):

            if i < train_n:
                row["split"] = "train"

            elif i < train_n + val_n:
                row["split"] = "val"

            else:
                row["split"] = "test"

        return pd.DataFrame(rows)

    def write(self, manifest_path):

        manifest = self.build()

        manifest_path = Path(manifest_path)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest.to_csv(manifest_path, index=False)

        return manifest_path


@app.command()
def build(
    config_path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
    )
):
    configure_logging()

    config = load_config(config_path)

    builder = ManifestBuilder(
        root=config["data"]["processed_dir"],
        splits=config["dataset"]["splits"],
        seed=config["project"]["seed"],
    )

    path = builder.write(config["data"]["manifest_path"])

    typer.echo(f"Manifest written to {path}")


def main():
    app()


if __name__ == "__main__":
    main()