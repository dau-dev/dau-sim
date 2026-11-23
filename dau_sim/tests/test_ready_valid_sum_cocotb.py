from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dau_sim.integrations.cocotb import run_cocotb_testbench
from dau_sim.tests.verilator_support import requires_cocotb_verilator

_SV_DIR = Path(__file__).parent / "sv"
_COCOTB_TEST_MODULE = "dau_sim.tests.cocotb_benches.ready_valid_sum_tb"
WAVEFORM_NAME = "dump.vcd"


@requires_cocotb_verilator
def test_ready_valid_sum_runs_real_cocotb_against_systemverilog(tmp_path: Path) -> None:
    results = run_cocotb_testbench(
        sources=(_SV_DIR / "ready_valid_sum.sv",),
        hdl_toplevel="ready_valid_sum",
        test_module=_COCOTB_TEST_MODULE,
        build_dir=tmp_path / "cocotb-verilator",
        results_xml=tmp_path / "cocotb-results.xml",
    )
    assert results.is_file()


@requires_cocotb_verilator
def test_waves_records_a_waveform_in_the_build_dir(tmp_path: Path) -> None:
    build_dir = tmp_path / "cocotb-verilator"
    run_cocotb_testbench(
        sources=(_SV_DIR / "ready_valid_sum.sv",),
        hdl_toplevel="ready_valid_sum",
        test_module=_COCOTB_TEST_MODULE,
        build_dir=build_dir,
        results_xml=tmp_path / "cocotb-results.xml",
        waves=True,
    )
    waveform = build_dir / WAVEFORM_NAME
    assert waveform.is_file() and waveform.stat().st_size > 0


@requires_cocotb_verilator
def test_testcase_runs_only_the_named_test(tmp_path: Path) -> None:
    results = run_cocotb_testbench(
        sources=(_SV_DIR / "ready_valid_sum.sv",),
        hdl_toplevel="ready_valid_sum",
        test_module=_COCOTB_TEST_MODULE,
        build_dir=tmp_path / "cocotb-verilator",
        results_xml=tmp_path / "cocotb-results.xml",
        testcase="reset_clears_a_held_result",
    )
    assert [case.get("name") for case in ET.parse(results).iter("testcase")] == ["reset_clears_a_held_result"]


def test_run_cocotb_testbench_validates_inputs(tmp_path: Path) -> None:
    """Argument validation happens before the runner is built, so it needs no Verilator."""
    with pytest.raises(ValueError, match="hdl_toplevel"):
        run_cocotb_testbench(sources=(_SV_DIR / "ready_valid_sum.sv",), hdl_toplevel="", test_module="x", build_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        run_cocotb_testbench(sources=(), hdl_toplevel="top", test_module="x", build_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_cocotb_testbench(sources=(tmp_path / "missing.sv",), hdl_toplevel="top", test_module="x", build_dir=tmp_path)
