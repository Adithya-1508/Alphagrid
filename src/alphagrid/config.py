from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_config(path=None) -> dict:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)
