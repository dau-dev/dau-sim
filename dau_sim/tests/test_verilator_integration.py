from __future__ import annotations

from pathlib import Path
from shutil import which

import pytest

from dau_sim.integrations.verilator import VerilatorUnavailableError, run_verilator_testbench

_SV_DIR = Path(__file__).parent / "sv"


def test_verilator_helper_reports_missing_tool(tmp_path: Path) -> None:
    with pytest.raises(VerilatorUnavailableError, match="verilator executable not found"):
        run_verilator_testbench(sources=(), top_module="missing_tool_tb", work_dir=tmp_path, verilator="definitely-not-verilator")


@pytest.mark.skipif(which("verilator") is None, reason="verilator not found")
def test_ready_valid_sum_systemverilog_testbench_runs_under_verilator(tmp_path: Path) -> None:
    result = run_verilator_testbench(
        sources=(
            _SV_DIR / "ready_valid_sum.sv",
            _SV_DIR / "ready_valid_sum_tb.sv",
        ),
        top_module="ready_valid_sum_tb",
        work_dir=tmp_path,
    )

    assert "READY_VALID_SUM_TB_OK" in result.stdout
