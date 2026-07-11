from __future__ import annotations
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)
