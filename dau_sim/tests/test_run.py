"""The run-sv task: a run is a composed CallableModel, not only a command."""

from pathlib import Path

from ccflow import NullContext

from dau_sim.config import run_request_config
from dau_sim.run import RunSvResult, RunSvTask

ADDER = """
module adder(
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] y
);
  assign y = a + b;
endmodule
""".strip()


def test_the_task_runs_a_design_and_reports_final_values(tmp_path: Path) -> None:
    src = tmp_path / "adder.sv"
    src.write_text(ADDER)
    result = RunSvTask(path=src, top="adder", cycles=1, inputs={"a": 40, "b": 2})(NullContext())
    assert isinstance(result, RunSvResult)
    assert result.module_name == "adder"
    assert result.latest["y"] == 42
    assert result.vcd_path is None


def test_the_task_writes_a_vcd_when_asked(tmp_path: Path) -> None:
    src = tmp_path / "adder.sv"
    src.write_text(ADDER)
    vcd = tmp_path / "adder.vcd"
    result = RunSvTask(path=src, top="adder", cycles=2, inputs={"a": 1, "b": 1}, vcd=vcd)(NullContext())
    assert result.vcd_path == vcd and vcd.is_file()
    assert "$timescale" in vcd.read_text()


def test_the_task_composes_from_the_config_tree(tmp_path: Path) -> None:
    src = tmp_path / "adder.sv"
    src.write_text(ADDER)
    result = run_request_config("task", "tasks/sim/run-sv", model_values={"path": src, "top": "adder", "cycles": 1, "inputs": {"a": 2, "b": 3}})
    assert result.latest["y"] == 5
