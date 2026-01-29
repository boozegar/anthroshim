"""Deprecated configuration module.

The server now uses a dedicated YAML model map file (model-map.yml) plus .env
for secrets.

This module is kept only to avoid breaking stale imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


Json = Dict[str, Any]


@dataclass
class AppConfig:
    data: Dict[str, Any] | None = None


def load_config(*args: Any, **kwargs: Any) -> AppConfig:
    raise RuntimeError("Deprecated: use model-map.yml")


def save_config(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Deprecated: use model-map.yml")


def update_config(*args: Any, **kwargs: Any) -> AppConfig:
    raise RuntimeError("Deprecated: use model-map.yml")


def config_to_public_dict(cfg: AppConfig) -> Json:
    return dict(cfg.data or {})
