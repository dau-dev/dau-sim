from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from ccflow import BaseModel
from hydra.errors import ConfigCompositionException
from pydantic import ConfigDict, field_validator

PACKAGE_URI_PREFIX = "package://"


class SimulationProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    sources: tuple[Path, ...]

    @field_validator("sources", mode="before")
    @classmethod
    def _resolve_sources(cls, sources: Any) -> tuple[Path, ...]:
        return tuple(_resolve_source(source) for source in sources)


def available_simulation_profiles(*, config_dir: str | None = None) -> tuple[str, ...]:
    from dau_sim.config import profile_names

    return profile_names(config_dir=config_dir)


def resolve_simulation_profile(name: str, *, config_dir: str | None = None) -> SimulationProfile:
    from dau_sim.config import load_config, profile_names

    known_names = profile_names(config_dir=config_dir)
    if name not in known_names:
        known = ", ".join(known_names)
        raise KeyError(f"unknown simulation profile {name!r}; expected one of: {known}")
    try:
        profile = load_config([f"profile=profiles/{name}"], overwrite=True, config_dir=config_dir)["profile"]
    except (ConfigCompositionException, KeyError):
        known = ", ".join(known_names)
        raise KeyError(f"unknown simulation profile {name!r}; expected one of: {known}") from None
    if not isinstance(profile, SimulationProfile):
        raise TypeError(f"profile config {name!r} did not produce a SimulationProfile")
    return profile


def _resolve_source(source: Path | str) -> Path:
    if isinstance(source, Path):
        return source
    if not source.startswith(PACKAGE_URI_PREFIX):
        return Path(source)
    resource = source.removeprefix(PACKAGE_URI_PREFIX)
    package, separator, resource_name = resource.partition("/")
    if not package or not separator or not resource_name:
        raise ValueError(f"invalid package resource URI: {source}")
    return Path(str(files(package).joinpath(resource_name)))
