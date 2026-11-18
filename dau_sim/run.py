"""Run one SystemVerilog or Verilog design for a number of cycles, as a task.

The composed form of the ``dau-sim run-sv`` command: the same fields the
command takes, as a ccflow ``CallableModel`` selected with
``task=tasks/sim/run-sv``, so a run can be configured, overridden and
recorded the way every other operation in the stack is. The command is a
thin front end over this.
"""

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from ccflow import CallableModel, Flow, NullContext, ResultBase
from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer

#: Annotate a ``Mapping`` field on a frozen model with this and it is stored
#: read-only and serialized as a plain dict.
ReadOnly = (AfterValidator(MappingProxyType), PlainSerializer(dict, return_type=dict))


class RunSvResult(ResultBase):
    """The final value of every signal after the run, and where the VCD went."""

    model_config = ConfigDict(frozen=True)

    module_name: str
    cycles: int
    latest: Annotated[Mapping[str, int], *ReadOnly]
    vcd_path: Path | None = None


class RunSvTask(CallableModel):
    """Parse ``path`` (top module ``top``, or the file's single module), run
    it for ``cycles`` clock cycles at ``clock_period_us`` with ``inputs``
    held, and report every signal's final value. With ``vcd`` set, the
    traces are written there at ``timescale``. Every field is required: the
    defaults are in the task config, ``task=tasks/sim/run-sv``."""

    model_config = ConfigDict(frozen=True)

    path: Path
    top: str | None
    cycles: int = Field(ge=1)
    clock_period_us: float = Field(gt=0)
    inputs: Annotated[Mapping[str, int], *ReadOnly] | None
    vcd: Path | None
    timescale: str

    def check(self) -> None:
        """The construction invariants, re-checked at use (model_copy skips them)."""
        if self.cycles < 1:
            raise ValueError(f"cycles must be at least 1, got {self.cycles}")
        if self.clock_period_us <= 0:
            raise ValueError(f"clock_period_us must be positive, got {self.clock_period_us}")

    @Flow.call
    def __call__(self, context: NullContext) -> RunSvResult:  # noqa: ARG002 (ccflow requires the name `context`)
        from dau_sim.api import Simulator

        self.check()
        sim = Simulator.from_sv_file(str(self.path), top=self.top)
        result = sim.run(cycles=self.cycles, clock_period=timedelta(microseconds=self.clock_period_us), inputs=dict(self.inputs or {}))
        if self.vcd is not None:
            sim.write_vcd(str(self.vcd), result, timescale=self.timescale)
        latest = {signal: value for signal, _ in sorted(result.traces) if (value := result.latest(signal)) is not None}
        return RunSvResult(module_name=result.module_name, cycles=self.cycles, latest=latest, vcd_path=self.vcd)
