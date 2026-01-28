from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


Json = Dict[str, Any]


@dataclass
class OpenAIConfig:
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    force_stream: bool = False
    image_url_object: bool = False


@dataclass
class AnthropicConfig:
    model_map: Dict[str, str] = None
    model_default: str | None = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str | None = None
    payloads: bool = False
    max_chars: int = 4000


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    ui_enabled: bool = True


@dataclass
class AppConfig:
    openai: OpenAIConfig
    anthropic: AnthropicConfig
    logging: LoggingConfig
    server: ServerConfig


ROOT_KEY = "api_transformer_config"


def default_config() -> AppConfig:
    return AppConfig(
        openai=OpenAIConfig(),
        anthropic=AnthropicConfig(model_map={}),
        logging=LoggingConfig(),
        server=ServerConfig(),
    )


def load_config(path: Path) -> AppConfig:
    cfg = default_config()
    if not path.exists():
        save_config(path, cfg)
        return cfg
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if raw.strip() else None
    if not isinstance(data, dict):
        return cfg
    data = data.get(ROOT_KEY, data)
    if not isinstance(data, dict):
        return cfg
    return _config_from_dict(data, cfg)


def save_config(path: Path, cfg: AppConfig) -> None:
    data = {ROOT_KEY: config_to_dict(cfg, include_secret=True)}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def update_config(cfg: AppConfig, updates: Json) -> AppConfig:
    if not isinstance(updates, dict):
        return cfg
    updates = updates.get(ROOT_KEY, updates)
    if not isinstance(updates, dict):
        return cfg

    updates = copy.deepcopy(updates)
    openai_updates = updates.get("openai")
    if isinstance(openai_updates, dict) and "api_key" in openai_updates:
        api_key = openai_updates.get("api_key")
        if api_key is None:
            pass
        elif isinstance(api_key, str) and not api_key.strip():
            openai_updates.pop("api_key", None)

    merged = _deep_merge(config_to_dict(cfg, include_secret=True), updates)
    return _config_from_dict(merged, cfg)


def config_to_public_dict(cfg: AppConfig) -> Json:
    data = config_to_dict(cfg, include_secret=True)
    openai = data.get("openai", {})
    openai["api_key_set"] = bool(cfg.openai.api_key)
    openai["api_key"] = ""
    return data


def config_to_dict(cfg: AppConfig, include_secret: bool) -> Json:
    openai = {
        "api_key": cfg.openai.api_key if include_secret else "",
        "base_url": cfg.openai.base_url,
        "force_stream": cfg.openai.force_stream,
        "image_url_object": cfg.openai.image_url_object,
    }
    anthropic = {
        "model_map": dict(cfg.anthropic.model_map or {}),
        "model_default": cfg.anthropic.model_default,
    }
    logging_cfg = {
        "level": cfg.logging.level,
        "file": cfg.logging.file,
        "payloads": cfg.logging.payloads,
        "max_chars": cfg.logging.max_chars,
    }
    server_cfg = {
        "host": cfg.server.host,
        "port": cfg.server.port,
        "ui_enabled": cfg.server.ui_enabled,
    }
    return {
        "openai": openai,
        "anthropic": anthropic,
        "logging": logging_cfg,
        "server": server_cfg,
    }


def _config_from_dict(data: Json, base: AppConfig) -> AppConfig:
    openai = _load_openai(data.get("openai"), base.openai)
    anthropic = _load_anthropic(data.get("anthropic"), base.anthropic)
    logging_cfg = _load_logging(data.get("logging"), base.logging)
    server_cfg = _load_server(data.get("server"), base.server)
    return AppConfig(
        openai=openai,
        anthropic=anthropic,
        logging=logging_cfg,
        server=server_cfg,
    )


def _load_openai(data: Any, base: OpenAIConfig) -> OpenAIConfig:
    if not isinstance(data, dict):
        return base
    return OpenAIConfig(
        api_key=_string_or_none(data.get("api_key"), base.api_key),
        base_url=_string_or_default(data.get("base_url"), base.base_url),
        force_stream=_bool_or_default(data.get("force_stream"), base.force_stream),
        image_url_object=_bool_or_default(data.get("image_url_object"), base.image_url_object),
    )


def _load_anthropic(data: Any, base: AnthropicConfig) -> AnthropicConfig:
    if not isinstance(data, dict):
        return base
    model_map = {}
    raw_map = data.get("model_map")
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            if isinstance(key, str) and isinstance(value, str) and key and value:
                model_map[key] = value
    return AnthropicConfig(
        model_map=model_map or dict(base.model_map or {}),
        model_default=_string_or_none(data.get("model_default"), base.model_default),
    )


def _load_logging(data: Any, base: LoggingConfig) -> LoggingConfig:
    if not isinstance(data, dict):
        return base
    level = _string_or_default(data.get("level"), base.level).upper()
    if not hasattr(logging, level):
        level = base.level
    return LoggingConfig(
        level=level,
        file=_string_or_none(data.get("file"), base.file),
        payloads=_bool_or_default(data.get("payloads"), base.payloads),
        max_chars=_int_or_default(data.get("max_chars"), base.max_chars),
    )


def _load_server(data: Any, base: ServerConfig) -> ServerConfig:
    if not isinstance(data, dict):
        return base
    return ServerConfig(
        host=_string_or_default(data.get("host"), base.host),
        port=_int_or_default(data.get("port"), base.port),
        ui_enabled=_bool_or_default(data.get("ui_enabled"), base.ui_enabled),
    )


def _deep_merge(base: Json, updates: Json) -> Json:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _string_or_default(value: Any, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _string_or_none(value: Any, default: str | None) -> str | None:
    if isinstance(value, str) and value:
        return value
    return default


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False
    return default


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
