from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dau_sim.api import Simulator
from dau_sim.perf import analyze_node_separation, benchmark_module, evaluate_delta

app = typer.Typer(help="dau-sim command line interface")
console = Console()


def _parse_kv_pairs(items: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid input '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(f"Invalid input '{item}'. Empty signal name.")
        try:
            parsed[name] = int(raw_value.strip(), 0)
        except ValueError as ex:
            raise typer.BadParameter(f"Invalid integer value in '{item}'.") from ex
    return parsed


@app.command("run-sv")
def run_sv(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="SystemVerilog or Verilog source file."),
    top: str | None = typer.Option(None, "--top", help="Top module name."),
    cycles: int = typer.Option(10, min=1, help="Number of cycles."),
    clock_period_us: float = typer.Option(1.0, "--clock-us", min=0.001, help="Clock period in microseconds."),
    inputs: list[str] = typer.Option([], "--input", "-i", help="Signal assignments, e.g. -i en=1 -i a=0x10."),
    vcd: Path | None = typer.Option(None, "--vcd", help="Optional output VCD path."),
    timescale: str = typer.Option("1ns", help="VCD timescale."),
) -> None:
    parsed_inputs = _parse_kv_pairs(inputs)
    sim = Simulator.from_sv_file(str(path), top=top)
    result = sim.run(cycles=cycles, clock_period=timedelta(microseconds=clock_period_us), inputs=parsed_inputs)

    if vcd is not None:
        sim.write_vcd(str(vcd), result, timescale=timescale)

    latest_table = Table(title="Latest signal values")
    latest_table.add_column("Signal")
    latest_table.add_column("Value", justify="right")

    for signal in sorted(result.traces.keys()):
        latest = result.latest(signal)
        if latest is not None:
            latest_table.add_row(signal, str(latest))

    console.print(f"Simulation completed for module '{result.module_name}'.")
    console.print(latest_table)
    if vcd is not None:
        console.print(f"Wrote VCD: {vcd}")


@app.command("perf-sv")
def perf_sv(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="SystemVerilog or Verilog source file."),
    top: str | None = typer.Option(None, "--top", help="Top module name."),
    cycles: int = typer.Option(30000, min=1, help="Cycles per measured run."),
    repeats: int = typer.Option(3, min=1, help="Measured repetitions."),
    warmup: int = typer.Option(1, min=0, help="Warmup runs per repetition."),
    inputs: list[str] = typer.Option([], "--input", "-i", help="Signal assignments, e.g. -i en=1."),
    amaranth_cps: float | None = typer.Option(None, "--amaranth-cps", min=0.0, help="Optional Amaranth baseline cycles/sec."),
    verilator_cps: float | None = typer.Option(None, "--verilator-cps", min=0.0, help="Optional Verilator baseline cycles/sec."),
) -> None:
    parsed_inputs = _parse_kv_pairs(inputs)
    sim = Simulator.from_sv_file(str(path), top=top)

    bench = benchmark_module(
        sim.module,
        cycles=cycles,
        repeats=repeats,
        warmup=warmup,
        inputs=parsed_inputs,
    )
    sep = analyze_node_separation(sim.module)
    delta = evaluate_delta(
        bench.cycles_per_second,
        amaranth_cycles_per_second=amaranth_cps,
        verilator_cycles_per_second=verilator_cps,
    )

    perf_table = Table(title="Performance")
    perf_table.add_column("Metric")
    perf_table.add_column("Value", justify="right")
    perf_table.add_row("compile median (s)", f"{bench.compile_seconds_median:.6f}")
    perf_table.add_row("run median (s)", f"{bench.run_seconds_median:.6f}")
    perf_table.add_row("dau-sim cycles/sec", f"{bench.cycles_per_second:,.2f}")
    perf_table.add_row("target cycles/sec (10x)", f"{bench.cycles_per_second * delta.multiplier_to_10x_current:,.2f}")
    if delta.vs_amaranth_ratio is not None:
        perf_table.add_row("vs amaranth", f"{delta.vs_amaranth_ratio:.2f}x")
    if delta.vs_verilator_ratio is not None:
        perf_table.add_row("vs verilator", f"{delta.vs_verilator_ratio:.4f}x")

    sep_table = Table(title="Node separation diagnostics")
    sep_table.add_column("Metric")
    sep_table.add_column("Value", justify="right")
    sep_table.add_row("comb blocks", str(sep.comb_blocks))
    sep_table.add_row("dependency edges", str(sep.dependency_edges))
    sep_table.add_row("connected components", str(sep.connected_components))
    sep_table.add_row("largest component", str(sep.largest_component))
    sep_table.add_row("singleton components", str(sep.singleton_components))

    console.print(perf_table)
    console.print(sep_table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
