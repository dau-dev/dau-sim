from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dau_core.hdl import (
    DAU_AGGREGATION_PKG_SV,
    DAU_INT32_ARROW_LITE_STREAM_AGGREGATION_SV,
    DAU_INT32_GROUPED_AGGREGATION_SV,
    DAU_INT32_MAP_ALU_SV,
    DAU_INT32_PREDICATE_FILTER_SV,
    DAU_INT32_RECORD_BATCH_AGGREGATION_SV,
    DAU_MAP_ALU_PKG_SV,
    DAU_PREDICATE_PKG_SV,
)


@dataclass(frozen=True)
class VerilatorProfile:
    name: str
    sources: tuple[Path, ...]
    top_module: str
    expect_stdout: str


_SV_TESTBENCH_DIR = Path(__file__).resolve().parents[1] / "tests" / "sv"
_DAU_CORE_SV_TESTBENCH_DIR = Path(str(DAU_INT32_RECORD_BATCH_AGGREGATION_SV)).resolve().parents[1] / "tests" / "sv"

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
    "dau-int32-arrow-lite-stream-aggregation": VerilatorProfile(
        name="dau-int32-arrow-lite-stream-aggregation",
        sources=(
            Path(str(DAU_AGGREGATION_PKG_SV)),
            Path(str(DAU_INT32_ARROW_LITE_STREAM_AGGREGATION_SV)),
            _DAU_CORE_SV_TESTBENCH_DIR / "dau_int32_arrow_lite_stream_aggregation_tb.sv",
        ),
        top_module="dau_int32_arrow_lite_stream_aggregation_tb",
        expect_stdout="DAU_INT32_ARROW_LITE_STREAM_AGGREGATION_TB_OK",
    ),
    "dau-int32-grouped-aggregation": VerilatorProfile(
        name="dau-int32-grouped-aggregation",
        sources=(
            Path(str(DAU_INT32_GROUPED_AGGREGATION_SV)),
            _DAU_CORE_SV_TESTBENCH_DIR / "dau_int32_grouped_aggregation_tb.sv",
        ),
        top_module="dau_int32_grouped_aggregation_tb",
        expect_stdout="DAU_INT32_GROUPED_AGGREGATION_TB_OK",
    ),
    "dau-int32-map-alu": VerilatorProfile(
        name="dau-int32-map-alu",
        sources=(
            Path(str(DAU_MAP_ALU_PKG_SV)),
            Path(str(DAU_INT32_MAP_ALU_SV)),
            _DAU_CORE_SV_TESTBENCH_DIR / "dau_int32_map_alu_tb.sv",
        ),
        top_module="dau_int32_map_alu_tb",
        expect_stdout="DAU_INT32_MAP_ALU_TB_OK",
    ),
    "dau-int32-predicate-filter": VerilatorProfile(
        name="dau-int32-predicate-filter",
        sources=(
            Path(str(DAU_PREDICATE_PKG_SV)),
            Path(str(DAU_INT32_PREDICATE_FILTER_SV)),
            _DAU_CORE_SV_TESTBENCH_DIR / "dau_int32_predicate_filter_tb.sv",
        ),
        top_module="dau_int32_predicate_filter_tb",
        expect_stdout="DAU_INT32_PREDICATE_FILTER_TB_OK",
    ),
    "dau-int32-record-batch-aggregation": VerilatorProfile(
        name="dau-int32-record-batch-aggregation",
        sources=(
            Path(str(DAU_AGGREGATION_PKG_SV)),
            Path(str(DAU_INT32_RECORD_BATCH_AGGREGATION_SV)),
            _DAU_CORE_SV_TESTBENCH_DIR / "dau_int32_record_batch_aggregation_tb.sv",
        ),
        top_module="dau_int32_record_batch_aggregation_tb",
        expect_stdout="DAU_INT32_RECORD_BATCH_AGGREGATION_TB_OK",
    ),
}


def available_verilator_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def register_verilator_profile(profile: VerilatorProfile, *, replace: bool = False) -> None:
    """Open registration: packages and user code add profiles without
    editing dau-sim (mirrors the dau-build config-overlay idiom)."""
    if not replace and profile.name in _PROFILES:
        raise ValueError(f"verilator profile {profile.name!r} is already registered; pass replace=True to override")
    _PROFILES[profile.name] = profile


def resolve_verilator_profile(name: str) -> VerilatorProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        known = ", ".join(available_verilator_profiles())
        raise KeyError(f"unknown verilator profile {name!r}; expected one of: {known}") from exc
