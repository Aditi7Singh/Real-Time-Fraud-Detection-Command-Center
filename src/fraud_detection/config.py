from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load YAML configuration and resolve project paths relative to the repository root."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    project = config.setdefault("project", {})
    root = Path(project.get("root_dir", Path.cwd())).resolve()
    project["root_dir"] = str(root)

    for key in ["data_dir", "processed_dir", "models_dir", "artifacts_dir"]:
        if key in project:
            project[key] = str((root / project[key]).resolve())

    if "mlflow_tracking_uri" in project:
        uri = str(project["mlflow_tracking_uri"])
        project["mlflow_tracking_uri"] = uri if "://" in uri else str((root / uri).resolve())

    return config


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default
