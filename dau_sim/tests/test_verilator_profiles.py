from __future__ import annotations

from pathlib import Path

import pytest

from dau_sim.integrations.verilator_profiles import available_verilator_profiles, resolve_verilator_profile


def test_available_verilator_profiles_lists_generic_benches() -> None:
    assert available_verilator_profiles() == ("ready-valid-sum",)


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
