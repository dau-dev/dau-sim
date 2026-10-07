"""Run one SystemVerilog or Verilog design for a number of cycles, as a task.

The composed form of the ``dau-sim run-sv`` command: the same fields the
command takes, as a ccflow ``CallableModel`` selected with
``task=tasks/sim/run-sv``, so a run can be configured, overridden and
recorded the way every other operation in the stack is. The command is a
thin front end over this.
"""

from datetime import timedelta
from pathlib import Path

from ccflow import CallableModel, Flow, NullContext, ResultBase


class RunSvResult(ResultBase):
    """The final value of every signal after the run, and where the VCD went."""

    module_name: str
    cycles: int
    latest: dict[str, int]
    vcd_path: Path | None = None


class RunSvTask(CallableModel):
    """Parse ``path`` (top module ``top``, or the file's single module), run
    it for ``cycles`` clock cycles at ``clock_period_us`` with ``inputs``
    held, and report every signal's final value. With ``vcd`` set, the
    traces are written there at ``timescale``."""

    path: Path
    top: str | None = None
    cycles: int = 10
    clock_period_us: float = 1.0
    inputs: dict[str, int] | None = None
    vcd: Path | None = None
    timescale: str = "1ns"

    @Flow.call
    def __call__(self, context: NullContext) -> RunSvResult:  # noqa: ARG002 (ccflow requires the name `context`)
        from dau_sim.api import Simulator

        if self.cycles < 1:
            raise ValueError(f"cycles must be at least 1, got {self.cycles}")
        if self.clock_period_us <= 0:
            raise ValueError(f"clock_period_us must be positive, got {self.clock_period_us}")
        sim = Simulator.from_sv_file(str(self.path), top=self.top)
        result = sim.run(cycles=self.cycles, clock_period=timedelta(microseconds=self.clock_period_us), inputs=self.inputs or {})
        if self.vcd is not None:
            sim.write_vcd(str(self.vcd), result, timescale=self.timescale)
        latest = {signal: value for signal, _ in sorted(result.traces) if (value := result.latest(signal)) is not None}
        return RunSvResult(module_name=result.module_name, cycles=self.cycles, latest=latest, vcd_path=self.vcd)
