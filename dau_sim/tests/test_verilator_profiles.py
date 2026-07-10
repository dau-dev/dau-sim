from __future__ import annotations

from pathlib import Path

import pytest

from dau_sim.integrations.verilator_profiles import (
    VerilatorProfile,
    available_verilator_profiles,
    register_verilator_profile,
    resolve_verilator_profile,
)


def test_available_verilator_profiles_lists_registered_benches() -> None:
    assert available_verilator_profiles() == (
        "dau-int32-arrow-lite-stream-aggregation",
        "dau-int32-grouped-aggregation",
        "dau-int32-map-alu",
        "dau-int32-predicate-filter",
        "dau-int32-record-batch-aggregation",
        "ready-valid-sum",
    )


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


def test_register_verilator_profile_open_registration() -> None:
    profile = VerilatorProfile(name="user-bench", sources=(Path("user_bench.sv"),), top_module="user_bench_tb", expect_stdout="USER_BENCH_OK")
    register_verilator_profile(profile)
    try:
        assert resolve_verilator_profile("user-bench") is profile
        assert "user-bench" in available_verilator_profiles()
        with pytest.raises(ValueError, match="already registered"):
            register_verilator_profile(profile)
        replacement = VerilatorProfile(name="user-bench", sources=(Path("v2.sv"),), top_module="v2_tb", expect_stdout="V2_OK")
        register_verilator_profile(replacement, replace=True)
        assert resolve_verilator_profile("user-bench") is replacement
    finally:
        from dau_sim.integrations.verilator_profiles import _PROFILES

        _PROFILES.pop("user-bench", None)
