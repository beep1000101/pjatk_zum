from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ConfigError(
            f"Failed to parse TOML config: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid TOML root (expected table/dict): {path}")
    return data


def require_table(cfg: dict[str, Any], key: str, *, path: Path) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ConfigError(
            f"Missing or invalid [{key}] table in config: {path}")
    return value


def require_str(table: dict[str, Any], key: str, *, path: Path, table_name: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"Missing or invalid '{key}' in [{table_name}] table in config: {path}"
        )
    return value


def optional_str(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    return value


def optional_int(table: dict[str, Any], key: str) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ConfigError(f"Invalid '{key}' (expected int)")
    return value


def optional_list_of_str(table: dict[str, Any], key: str) -> list[str] | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ConfigError(f"Invalid '{key}' (expected list[str])")
    return value


def as_path(value: str) -> Path:
    return Path(value).expanduser()
