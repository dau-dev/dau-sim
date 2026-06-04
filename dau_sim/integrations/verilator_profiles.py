from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerilatorProfile:
    name: str
    sources: tuple[Path, ...]
    top_module: str
    expect_stdout: str


_SV_TESTBENCH_DIR = Path(__file__).resolve().parents[1] / "tests" / "sv"

_PROFILES: dict[str, VerilatorProfile] = {
    "ready-valid-sum": VerilatorProfile(
        name="ready-valid-sum",
        sources=(
            _SV_TESTBENCH_DIR / "ready_valid_sum.sv",
            _SV_TESTBENCH_DIR / "ready_valid_sum_tb.sv",
        ),
        top_module="ready_valid_sum_tb",
        expect_stdout="READY_VALID_SUM_TB_OK",
    ),
}


def available_verilator_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def resolve_verilator_profile(name: str) -> VerilatorProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        known = ", ".join(available_verilator_profiles())
        raise KeyError(f"unknown verilator profile {name!r}; expected one of: {known}") from exc
