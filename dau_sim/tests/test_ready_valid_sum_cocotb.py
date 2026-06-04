from __future__ import annotations

from pathlib import Path
from shutil import which

import pytest
from cocotb_tools.runner import get_runner

_SV_DIR = Path(__file__).parent / "sv"
_COCOTB_TEST_MODULE = "dau_sim.tests.cocotb_benches.ready_valid_sum_tb"


@pytest.mark.skipif(which("verilator") is None, reason="verilator not found")
def test_ready_valid_sum_runs_real_cocotb_against_systemverilog(tmp_path: Path) -> None:
    runner = get_runner("verilator")
    build_dir = tmp_path / "cocotb-verilator"
    runner.build(
        sources=(_SV_DIR / "ready_valid_sum.sv",),
        hdl_toplevel="ready_valid_sum",
        build_dir=build_dir,
        always=True,
        build_args=("--timing", "-Wno-fatal"),
    )
    runner.test(
        hdl_toplevel="ready_valid_sum",
        test_module=_COCOTB_TEST_MODULE,
        build_dir=build_dir,
        results_xml=str(tmp_path / "cocotb-results.xml"),
    )
