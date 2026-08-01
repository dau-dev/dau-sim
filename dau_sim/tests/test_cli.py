from pathlib import Path

import typer
from typer.testing import CliRunner

from dau_sim.cli import _parse_kv_pairs, app


def test_parse_kv_pairs_supports_base_prefixes() -> None:
    parsed = _parse_kv_pairs(["a=10", "b=0x10", "c=0b11"])
    assert parsed == {"a": 10, "b": 16, "c": 3}


def test_parse_kv_pairs_rejects_bad_items() -> None:
    try:
        _parse_kv_pairs(["broken"])
    except typer.BadParameter as ex:
        assert "Expected NAME=VALUE" in str(ex)
    else:
        raise AssertionError("Expected parse failure")


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-sv" in result.stdout
    assert "perf-sv" in result.stdout


def test_run_sv_command(tmp_path: Path) -> None:
    src = tmp_path / "adder.sv"
    src.write_text(
        """
module adder(
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] y
);
  assign y = a + b;
endmodule
""".strip()
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run-sv", str(src), "--top", "adder", "--cycles", "1", "-i", "a=40", "-i", "b=2"])

    assert result.exit_code == 0
    assert "Simulation completed" in result.stdout


def test_perf_sv_command_invokes_composed_task(tmp_path: Path, monkeypatch) -> None:
    from dau_sim.perf import BenchmarkResult, NodeSeparationStats, PerformanceDelta, PerfSvResult

    src = tmp_path / "adder.sv"
    src.write_text(
        """
module adder(
    input logic [7:0] a,
    input logic [7:0] b,
    output logic [7:0] y
);
    assign y = a + b;
endmodule
""".strip()
    )

    captured = {}

    def fake_run_request_config(request_kind, request_name, *, model_values, **kwargs):
        captured.update(request_kind=request_kind, request_name=request_name, model_values=model_values, kwargs=kwargs)
        return PerfSvResult(
            benchmark=BenchmarkResult(compile_seconds_median=0.1, run_seconds_median=0.2, cycles_per_second=50.0),
            node_separation=NodeSeparationStats(
                comb_blocks=1,
                dependency_edges=0,
                connected_components=1,
                largest_component=1,
                singleton_components=1,
            ),
            delta=PerformanceDelta(
                dau_cycles_per_second=50.0,
                vs_amaranth_ratio=None,
                vs_verilator_ratio=None,
                multiplier_to_10x_current=10.0,
            ),
        )

    monkeypatch.setattr("dau_sim.config.run_request_config", fake_run_request_config)
    runner = CliRunner()
    result = runner.invoke(app, ["perf-sv", str(src), "--top", "adder", "--cycles", "10", "--repeats", "1", "--warmup", "0"])

    assert result.exit_code == 0
    assert captured["request_kind"] == "task"
    assert captured["request_name"] == "tasks/analysis/perf-sv"
    assert captured["model_values"]["path"] == src
    assert "dau-sim cycles/sec" in result.stdout
    assert "Node separation diagnostics" in result.stdout
