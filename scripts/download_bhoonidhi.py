"""Download Bhoonidhi STAC products from an official JSON query."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from common.config import load_config
from common.logging import configure_logging
from datasets.bhoonidhi_client import BhoonidhiClient, BhoonidhiSettings

app = typer.Typer(help="Download Bhoonidhi products using official API endpoints.")


@app.command()
def download(
    query_path: Path = typer.Option(..., "--query", help="Path to a Bhoonidhi STAC search JSON."),
    output_dir: Path = typer.Option(Path("data/raw/bhoonidhi"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/default.yaml"), "--config"),
    max_pages: int | None = typer.Option(None, "--max-pages"),
) -> None:
    """Search Bhoonidhi and download online products returned by the query."""
    configure_logging()
    config = load_config(config_path)
    query = json.loads(query_path.read_text(encoding="utf-8"))
    if "filter" not in query:
        query["filter"] = {"args": [{"property": "Online"}, "Y"], "op": "eq"}
        query["filter-lang"] = "cql2-json"
    with BhoonidhiClient(BhoonidhiSettings.from_config(config)) as client:
        items = client.search_all(query, max_pages=max_pages)
        downloaded = client.download_many(items, output_dir)
    typer.echo(f"Downloaded {len(downloaded)} product(s) to {output_dir}")


def main() -> None:
    """Run the Typer CLI entry point."""
    app()


if __name__ == "__main__":
    main()
