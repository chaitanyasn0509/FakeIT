"""Configuration loading helpers for YAML files with environment interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


class ConfigError(RuntimeError):
    """Raised when project configuration cannot be loaded or resolved."""


def _interpolate_env(value: Any) -> Any:
    """Replace ${VAR:default} markers in YAML values with environment variables."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2)
            return os.environ.get(key, default or "")

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate_env(item) for key, item in value.items()}
    return value


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file and interpolate environment variables."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return _interpolate_env(data)


def deep_get(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested value from a dictionary using dot notation."""
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
