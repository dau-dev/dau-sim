from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AMARANTH_REPO = REPO_ROOT / "amaranth"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AMARANTH_REPO) not in sys.path:
    sys.path.insert(0, str(AMARANTH_REPO))


_VERILATOR_RUNTIME_CACHE: dict[int, tuple[Path, Path]] = {}
CROSS_SIM_CYCLES = int(os.getenv("DAU_BENCH_CYCLES", "5000"))


@atexit.register
def _cleanup_verilator_runtime_cache() -> None:
    for td, _exe in _VERILATOR_RUNTIME_CACHE.values():
        shutil.rmtree(td, ignore_errors=True)


def _make_counter():
    from amaranth import Elaboratable, Module as AModule, Signal

    class Counter(Elaboratable):
        def __init__(self):
            self.en = Signal(1)
            self.count = Signal(32)

        def elaborate(self, platform):
            m = AModule()
            with m.If(self.en):
                m.d.sync += self.count.eq(self.count + 1)
            return m

    return Counter


def _run_dau_sim(cycles: int) -> None:
    from dau_sim.compiler import compile_module
    from dau_sim.frontends import from_amaranth

    Counter = _make_counter()
    dut = Counter()
    ir_mod = from_amaranth(dut)
    cm = compile_module(ir_mod)
    cm.run(cycles=cycles, inputs={"en": 1}, trace_signals=["count"], return_traces=False)


def _run_amaranth_sim(cycles: int) -> None:
    from amaranth.sim import Simulator, Tick

    Counter = _make_counter()
    dut = Counter()
    sim = Simulator(dut)
    sim.add_clock(1e-6)

    def proc():
        yield dut.en.eq(1)
        for _ in range(cycles):
            yield Tick()

    sim.add_process(proc)
    sim.run()


def _run_amaranth_cxxsim(cycles: int) -> None:
    from amaranth.sim import Simulator, Tick

    Counter = _make_counter()
    dut = Counter()
    sim = Simulator(dut, engine="cxxsim")

    sim.add_clock(1e-6)

    def proc():
        yield dut.en.eq(1)
        for _ in range(cycles):
            yield Tick()

    sim.add_process(proc)
    sim.run()


def _compile_verilator_binary(cycles: int, *, persist: bool = False) -> tuple[Path, Path]:
    verilog = textwrap.dedent(
        f"""
        `timescale 1ns/1ps
        module counter(
            input wire clk,
            input wire rst,
            input wire en,
            output reg [31:0] count
        );
          always @(posedge clk) begin
            if (rst) count <= 32'd0;
            else if (en) count <= count + 32'd1;
          end
        endmodule

        module tb;
          reg clk = 0;
          reg rst = 1;
          reg en = 1;
          wire [31:0] count;
          integer i;

          counter dut(
            .clk(clk),
            .rst(rst),
            .en(en),
            .count(count)
          );

          initial begin
            #1 rst = 0;
            for (i = 0; i < {cycles}; i = i + 1) begin
              #1 clk = 1;
              #1 clk = 0;
            end
            $finish;
          end
        endmodule
        """
    )

    if persist:
        td = Path(tempfile.mkdtemp(prefix="dau_sim_verilator_rt_"))
    else:
        td = Path(tempfile.mkdtemp(prefix="dau_sim_verilator_"))
    src = td / "tb.v"
    src.write_text(verilog)
    # the canonical runner is the single Verilator invocation path; the bench
    # reuses its compiled executable for repeat runs
    from dau_sim.integrations.verilator import VerilatorExecutionError, run_verilator_testbench

    try:
        result = run_verilator_testbench(sources=(src,), top_module="tb", work_dir=td)
    except VerilatorExecutionError as exc:
        shutil.rmtree(td, ignore_errors=True)
        raise RuntimeError(f"verilator compile/run failed: {exc}") from exc
    return td, result.executable_path


def _run_verilator_compiled(exe: Path) -> None:
    rp = subprocess.run([str(exe)], cwd=exe.parent.parent, capture_output=True, text=True)
    if rp.returncode != 0:
        raise RuntimeError("verilator run failed")


def _run_verilator(cycles: int) -> None:
    td, exe = _compile_verilator_binary(cycles, persist=False)
    try:
        _run_verilator_compiled(exe)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _run_verilator_runtime(cycles: int) -> None:
    if cycles not in _VERILATOR_RUNTIME_CACHE:
        _VERILATOR_RUNTIME_CACHE[cycles] = _compile_verilator_binary(cycles, persist=True)
    _td, exe = _VERILATOR_RUNTIME_CACHE[cycles]
    _run_verilator_compiled(exe)


def _prepare_verilator_runtime(cycles: int) -> None:
    if cycles not in _VERILATOR_RUNTIME_CACHE:
        _VERILATOR_RUNTIME_CACHE[cycles] = _compile_verilator_binary(cycles, persist=True)


def _ensure_backend_available(backend: str) -> None:
    if backend in {"dau-sim", "amaranth-sim", "cxxsim"}:
        try:
            _make_counter()
            if backend == "cxxsim":
                import inspect

                from amaranth.sim import Simulator

                if "cxxsim" not in inspect.getsource(Simulator.__init__):
                    pytest.skip("cxxsim unavailable: this Amaranth build exposes only pysim")
        except Exception as exc:
            pytest.skip(f"{backend} unavailable: {exc}")

    if backend in {"verilator-compile-run", "verilator-runtime"} and shutil.which("verilator") is None:
        pytest.skip("verilator unavailable: not found on PATH")


@pytest.mark.parametrize(
    "backend,runner",
    [
        ("dau-sim", _run_dau_sim),
        ("amaranth-sim", _run_amaranth_sim),
        ("cxxsim", _run_amaranth_cxxsim),
        ("verilator-compile-run", _run_verilator),
        ("verilator-runtime", _run_verilator_runtime),
    ],
)
def test_benchmark_cross_simulators(benchmark, backend: str, runner) -> None:
    _ensure_backend_available(backend)
    if backend == "verilator-runtime":
        # Compile once outside the measured region; benchmark only executable runtime.
        _prepare_verilator_runtime(CROSS_SIM_CYCLES)
    benchmark.name = f"cross_simulators.{backend}"
    benchmark(lambda: runner(CROSS_SIM_CYCLES))
