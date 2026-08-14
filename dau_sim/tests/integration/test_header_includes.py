"""A source that includes a header beside it must actually compile.

This is the real proof of the include-path contract, and it needs
verilator. It lives here because ``dau_sim/tests/integration/`` is omitted
from coverage: a test that cannot run without a toolchain would otherwise
report its own body as uncovered on every runner that lacks one, which is
every CI runner here.

The unit-level companion in ``tests/test_cocotb_includes.py`` stands in for
the runner and checks the same contract everywhere the suite runs.
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

import pytest

from dau_sim.integrations.cocotb import run_cocotb_testbench

_HEADER = """\
`ifndef WIDTH_SVH
`define WIDTH_SVH
`define SUM_WIDTH 8
`endif
"""

_MODULE = """\
`include "width.svh"
module included_header_top (
    input  wire                    clk,
    input  wire [`SUM_WIDTH-1:0]   a,
    output reg  [`SUM_WIDTH-1:0]   y
);
    always @(posedge clk) y <= a + 1'b1;
endmodule
"""

_TEST_MODULE = """\
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def increments(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.a.value = 41
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert int(dut.y.value) == 42, f"header-defined width did not build: {int(dut.y.value)}"
"""


@pytest.mark.skipif(which("verilator") is None, reason="verilator not found")
def test_a_header_beside_its_source_is_found(tmp_path: Path, monkeypatch) -> None:
    """Without the include path this raises CalledProcessError from verilator."""
    hdl = tmp_path / "hdl"
    hdl.mkdir()
    (hdl / "width.svh").write_text(_HEADER)
    (hdl / "included_header_top.sv").write_text(_MODULE)

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "included_header_tb.py").write_text(_TEST_MODULE)
    monkeypatch.syspath_prepend(str(tests))

    results = run_cocotb_testbench(
        sources=(hdl / "included_header_top.sv",),
        hdl_toplevel="included_header_top",
        test_module="included_header_tb",
        build_dir=tmp_path / "build",
    )
    assert results.exists()
