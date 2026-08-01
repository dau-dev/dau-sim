from __future__ import annotations

from dau_sim.integrations.profiles import SimulationProfile, available_simulation_profiles, resolve_simulation_profile


class VerilatorProfile(SimulationProfile):
    top_module: str
    expect_stdout: str


def available_verilator_profiles(*, config_dir: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name in available_simulation_profiles(config_dir=config_dir)
        if isinstance(resolve_simulation_profile(name, config_dir=config_dir), VerilatorProfile)
    )


def resolve_verilator_profile(name: str, *, config_dir: str | None = None) -> VerilatorProfile:
    try:
        profile = resolve_simulation_profile(name, config_dir=config_dir)
    except KeyError:
        known = ", ".join(available_verilator_profiles(config_dir=config_dir))
        raise KeyError(f"unknown verilator profile {name!r}; expected one of: {known}") from None
    if not isinstance(profile, VerilatorProfile):
        raise TypeError(f"simulation profile {name!r} is not a VerilatorProfile")
    return profile
