from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def model_id_for_seed(config: dict[str, Any], seed: str | None = None) -> str:
    models = config.get("models") or {}
    if seed:
        if seed not in models:
            raise KeyError(f"Unknown seed {seed!r}; available: {sorted(models)}")
        return str(models[seed])
    if "model_name" in config:
        return str(config["model_name"])
    if models:
        return str(next(iter(models.values())))
    raise KeyError("Config must define model_name or models")


def safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_").replace(" ", "_")


@dataclass
class ModelLoadConfig:
    model_name: str
    device: str = "cuda"
    dtype: str = "bf16"
    revision: str | None = None
    trust_remote_code: bool = False


def model_load_config(config: dict[str, Any], model_name: str) -> ModelLoadConfig:
    return ModelLoadConfig(
        model_name=model_name,
        device=str(config.get("device", "cuda")),
        dtype=str(config.get("dtype", "bf16")),
        revision=config.get("model_revision_or_seed"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
