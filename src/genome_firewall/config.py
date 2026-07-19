"""Configuration loading shared by command-line pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedConfig:
    """Parsed configuration together with its filesystem context."""

    values: dict[str, Any]
    path: Path
    project_root: Path

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        try:
            from ruamel.yaml import YAML
        except ImportError as exc:
            raise RuntimeError(
                "A YAML parser is required. Install project dependencies or PyYAML."
            ) from exc
        with path.open("r", encoding="utf-8") as handle:
            parsed = YAML(typ="safe").load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)

    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return parsed


def load_config(path: str | Path) -> LoadedConfig:
    """Load YAML and treat the parent of ``configs/`` as the project root."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    project_root = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else config_path.parent
    )
    return LoadedConfig(
        values=_read_yaml(config_path),
        path=config_path,
        project_root=project_root,
    )
