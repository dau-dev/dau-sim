from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ccflow import ModelRegistry
from ccflow.utils.hydra import ConfigLoadResult, cfg_run, load_config as base_load_config
from omegaconf import OmegaConf

__all__ = ("compose_config", "load_config", "profile_names", "request_config", "run_request_config")


def load_config(
    overrides: Sequence[str] | None = None,
    *,
    overwrite: bool = False,
    config_dir: str | None = None,
    version_base: str | None = None,
) -> ModelRegistry:
    result = _load_base_config(overrides, config_dir=config_dir, version_base=version_base)
    registry = ModelRegistry.root()
    registry.load_config(result.cfg, overwrite=overwrite)
    return registry


def request_config(
    request_kind: str,
    request_name: str,
    *,
    model_values: Mapping[str, Any] | None = None,
    overrides: Sequence[str] | None = None,
    config_dir: str | None = None,
    version_base: str | None = None,
) -> ConfigLoadResult:
    result = _load_base_config(
        (f"{request_kind}={request_name}", *(overrides or ())),
        config_dir=config_dir,
        version_base=version_base,
    )
    for key, value in (model_values or {}).items():
        OmegaConf.update(result.cfg, f"model.{key}", _config_value(value), merge=False, force_add=True)
    return result


def run_request_config(
    request_kind: str,
    request_name: str,
    *,
    model_values: Mapping[str, Any] | None = None,
    overrides: Sequence[str] | None = None,
    config_dir: str | None = None,
    version_base: str | None = None,
):
    return cfg_run(
        request_config(
            request_kind,
            request_name,
            model_values=model_values,
            overrides=overrides,
            config_dir=config_dir,
            version_base=version_base,
        ).cfg
    )


def compose_config(
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | None = None,
    version_base: str | None = None,
) -> ConfigLoadResult:
    return _load_base_config(overrides, config_dir=config_dir, version_base=version_base)


def profile_names(*, config_dir: str | None = None, version_base: str | None = None) -> tuple[str, ...]:
    result = _load_base_config(config_dir=config_dir, version_base=version_base, debug=True)
    return tuple(sorted(result.group_options.get("profile/profiles", ())))


def _load_base_config(
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | None = None,
    version_base: str | None = None,
    debug: bool = False,
) -> ConfigLoadResult:
    parent_dir = str(Path(__file__).resolve().parent)
    return base_load_config(
        root_config_dir=parent_dir,
        root_config_name="base",
        config_dir=config_dir,
        overrides=list(overrides or ()),
        version_base=version_base,
        basepath=parent_dir,
        debug=debug,
    )


def _config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_config_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _config_value(item) for key, item in value.items()}
    return value
