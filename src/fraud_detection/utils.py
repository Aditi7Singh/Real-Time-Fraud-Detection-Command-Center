from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_divide(numerator: pd.Series | np.ndarray | float, denominator: pd.Series | np.ndarray | float, default: float = 0.0):
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        return np.where(np.asarray(denominator) != 0, np.asarray(numerator) / np.asarray(denominator), default)
    return numerator / denominator if denominator != 0 else default
