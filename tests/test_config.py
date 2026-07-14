"""Tests for shared configuration helpers."""

from __future__ import annotations

from pathlib import Path

from common.config import deep_get, load_config


def test_load_config_interpolates_environment(tmp_path: Path, monkeypatch) -> None:
    """Environment markers in YAML are replaced with runtime values."""
    monkeypatch.setenv("EXAMPLE_VALUE", "resolved")
    path = tmp_path / "config.yaml"
    path.write_text("section:\n  value: ${EXAMPLE_VALUE:default}\n", encoding="utf-8")
    config = load_config(path)
    assert deep_get(config, "section.value") == "resolved"


def test_deep_get_returns_default_for_missing_path() -> None:
    """Missing nested keys return the supplied default."""
    assert deep_get({"a": {"b": 1}}, "a.c", default=5) == 5
