from pathlib import Path

from typer.testing import CliRunner

from dau_sim.cli import _parse_kv_pairs, app


def test_parse_kv_pairs_supports_base_prefixes() -> None:
    parsed = _parse_kv_pairs(["a=10", "b=0x10", "c=0b11"])
    assert parsed == {"a": 10, "b": 16, "c": 3}


def test_parse_kv_pairs_rejects_bad_items() -> None:
    try:
        _parse_kv_pairs(["broken"])
    except Exception as ex:  # typer.BadParameter
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


def test_perf_sv_command(tmp_path: Path) -> None:
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
    result = runner.invoke(app, ["perf-sv", str(src), "--top", "adder", "--cycles", "10", "--repeats", "1", "--warmup", "0"])

    assert result.exit_code == 0
    assert "dau-sim cycles/sec" in result.stdout
    assert "Node separation diagnostics" in result.stdout
