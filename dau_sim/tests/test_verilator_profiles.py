from __future__ import annotations

from pathlib import Path

import pytest
from ccflow import BaseModel, ModelRegistry

from dau_sim.integrations.cocotb import CocotbProfile
from dau_sim.integrations.verilator_profiles import (
    VerilatorProfile,
    available_verilator_profiles,
    resolve_verilator_profile,
)


def test_available_verilator_profiles_lists_registered_benches() -> None:
    assert available_verilator_profiles() == ("ready-valid-sum",)


def test_verilator_profile_is_a_ccflow_model() -> None:
    assert issubclass(VerilatorProfile, BaseModel)
    assert issubclass(CocotbProfile, BaseModel)


def test_python_literal_profile_registry_is_removed() -> None:
    import dau_sim.integrations.verilator_profiles as profiles

    assert not hasattr(profiles, "_PROFILES")
    assert not hasattr(profiles, "register_verilator_profile")


@pytest.mark.parametrize("name", available_verilator_profiles())
def test_resolve_verilator_profile_returns_existing_sources(name: str) -> None:
    profile = resolve_verilator_profile(name)

    assert profile.name == name
    assert profile.top_module
    assert profile.expect_stdout
    assert profile.sources
    for source in profile.sources:
        assert Path(source).is_file()


def test_resolve_verilator_profile_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown verilator profile"):
        resolve_verilator_profile("unknown-profile")


def test_profile_config_group_loads_through_model_registry() -> None:
    from dau_sim.config import load_config

    registry = load_config(["profile=profiles/ready-valid-sum"], overwrite=True)

    assert isinstance(registry, ModelRegistry)
    assert isinstance(registry["profile"], VerilatorProfile)


def test_profile_config_overlay_provides_open_registration(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    profile_dir = config_dir / "profile" / "profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "user-bench.yaml").write_text(
        """# @package profile
_target_: dau_sim.integrations.verilator_profiles.VerilatorProfile
name: user-bench
sources: [user_bench.sv]
top_module: user_bench_tb
expect_stdout: USER_BENCH_OK
""",
        encoding="utf-8",
    )

    profile = resolve_verilator_profile("user-bench", config_dir=str(config_dir))

    assert profile.name == "user-bench"
    assert profile.top_module == "user_bench_tb"
