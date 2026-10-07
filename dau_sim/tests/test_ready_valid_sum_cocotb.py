from __future__ import annotations

from pathlib import Path

import pytest

from dau_sim.integrations.cocotb import run_cocotb_testbench
from dau_sim.tests.verilator_support import requires_cocotb_verilator

_SV_DIR = Path(__file__).parent / "sv"
_COCOTB_TEST_MODULE = "dau_sim.tests.cocotb_benches.ready_valid_sum_tb"


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


def test_run_cocotb_testbench_validates_inputs(tmp_path: Path) -> None:
    """Argument validation happens before the runner is built, so it needs no Verilator."""
    with pytest.raises(ValueError, match="hdl_toplevel"):
        run_cocotb_testbench(sources=(_SV_DIR / "ready_valid_sum.sv",), hdl_toplevel="", test_module="x", build_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        run_cocotb_testbench(sources=(), hdl_toplevel="top", test_module="x", build_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_cocotb_testbench(sources=(tmp_path / "missing.sv",), hdl_toplevel="top", test_module="x", build_dir=tmp_path)
