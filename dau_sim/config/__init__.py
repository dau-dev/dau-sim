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
    overrides: Sequence[str] | None = None,
    config_dir: str | None = None,
    version_base: str | None = None,
) -> ConfigLoadResult:
    """Compose ``<request_kind>=<request_name>`` with Hydra overrides. Task
    fields are set the Hydra way (``model.<field>=<value>``), so a request
    composed here is the same request a user types on the command line."""
    return _load_base_config(
        (f"{request_kind}={request_name}", *(overrides or ())),
        config_dir=config_dir,
        version_base=version_base,
    )


def model_overrides(values: Mapping[str, Any]) -> list[str]:
    """``model.<field>=<value>`` overrides for task fields, in Hydra's
    override grammar: strings and paths quoted, ``None`` as ``null``,
    mappings as ``{key:value,...}``."""
    return [f"model.{key}={_override_value(value)}" for key, value in values.items()]


def _override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Path | str):
        text = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{text}'"
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{key}:{_override_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, tuple | list):
        return "[" + ",".join(_override_value(item) for item in value) + "]"
    raise TypeError(f"cannot express {type(value).__name__} as a Hydra override")


def run_request_config(
    request_kind: str,
    request_name: str,
    *,
    overrides: Sequence[str] | None = None,
    config_dir: str | None = None,
    version_base: str | None = None,
):
    return cfg_run(request_config(request_kind, request_name, overrides=overrides, config_dir=config_dir, version_base=version_base).cfg)


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
