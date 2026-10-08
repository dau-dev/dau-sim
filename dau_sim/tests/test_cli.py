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

    assert result.exit_code == 0, result.stdout
    assert "Simulation completed" in result.stdout
    assert "42" in result.stdout


def test_run_sv_command_invokes_composed_task(tmp_path: Path, monkeypatch) -> None:
    """The command is a front end over ``task=tasks/sim/run-sv``: every option
    lands in the task model, so a run is configured the same way whether it
    comes from the shell or from a composed config."""
    from dau_sim.run import RunSvResult

    src = tmp_path / "adder.sv"
    src.write_text("module adder(input logic [7:0] a, output logic [7:0] y); assign y = a; endmodule")
    captured: dict[str, object] = {}

    def fake_run_request_config(request_kind, request_name, *, overrides=None, **_kwargs):
        captured["request_kind"] = request_kind
        captured["request_name"] = request_name
        captured["overrides"] = list(overrides)
        return RunSvResult(module_name="adder", cycles=3, latest={"y": 7}, vcd_path=tmp_path / "out.vcd")

    monkeypatch.setattr("dau_sim.config.run_request_config", fake_run_request_config)
    result = CliRunner().invoke(app, ["run-sv", str(src), "--top", "adder", "--cycles", "3", "-i", "a=7", "--vcd", str(tmp_path / "out.vcd")])

    assert result.exit_code == 0, result.stdout
    assert captured["request_kind"] == "task"
    assert captured["request_name"] == "tasks/sim/run-sv"
    assert f"model.path='{src}'" in captured["overrides"]
    assert "model.inputs={a:7}" in captured["overrides"]
    assert "model.cycles=3" in captured["overrides"]
    assert "model.top='adder'" in captured["overrides"]
    assert "Wrote VCD" in result.stdout


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

    def fake_run_request_config(request_kind, request_name, *, overrides, **kwargs):
        captured.update(request_kind=request_kind, request_name=request_name, overrides=list(overrides), kwargs=kwargs)
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
    assert f"model.path='{src}'" in captured["overrides"]
    assert "model.amaranth_cycles_per_second=null" in captured["overrides"]
    assert "dau-sim cycles/sec" in result.stdout
    assert "Node separation diagnostics" in result.stdout


def test_a_relative_config_dir_resolves_against_the_callers_directory(tmp_path: Path, monkeypatch) -> None:
    """The config loader searches a relative directory under the installed
    package, so the CLI hands it an absolute path."""
    from dau_sim.run import RunSvResult

    src = tmp_path / "adder.sv"
    src.write_text("module adder(input logic [7:0] a, output logic [7:0] y); assign y = a; endmodule")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    captured: dict[str, object] = {}

    def fake_run_request_config(request_kind, request_name, *, overrides=None, config_dir=None, **_kwargs):
        captured["config_dir"] = config_dir
        return RunSvResult(module_name="adder", cycles=1, latest={"y": 0}, vcd_path=None)

    monkeypatch.setattr("dau_sim.config.run_request_config", fake_run_request_config)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["run-sv", str(src), "--top", "adder", "--config-dir", "overlay"])

    assert result.exit_code == 0, result.stdout
    assert captured["config_dir"] == str(overlay.resolve())


def test_the_cli_composes_the_same_request_a_user_would_type(tmp_path: Path) -> None:
    """The overrides the CLI builds are Hydra overrides: composing them
    yields the task with every field set, including a quoted path, a
    mapping of inputs and a null."""
    from dau_sim.config import model_overrides, request_config

    src = tmp_path / "a design's file.sv"
    overrides = model_overrides({"path": src, "top": None, "cycles": 4, "inputs": {"en": 1, "a": 16}, "vcd": None, "timescale": "1ns"})
    model = request_config("task", "tasks/sim/run-sv", overrides=overrides).cfg.model
    assert (model.path, model.top, model.cycles, dict(model.inputs), model.vcd, model.timescale) == (
        str(src),
        None,
        4,
        {"en": 1, "a": 16},
        None,
        "1ns",
    )
