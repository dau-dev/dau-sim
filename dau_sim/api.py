from datetime import timedelta

from ccflow import BaseModel
from pydantic import ConfigDict

from dau_sim.compiler import compile_module
from dau_sim.compiler.compile import CompiledModule
from dau_sim.ir.module import Module


class SimulationResult(BaseModel):
    """Container for simulation output with interactive helpers."""

    model_config = ConfigDict(frozen=True)

    module_name: str
    traces: tuple[tuple[str, tuple[tuple[object, int], ...]], ...]

    def latest(self, signal: str) -> int | None:
        """Return the most recent value for a signal, if available."""
        values = dict(self.traces).get(signal, ())
        if not values:
            return None
        return values[-1][1]

    def to_rows(self, *, signals: list[str] | None = None) -> list[dict[str, object]]:
        """Return trace points as row dictionaries for notebooks/data tools."""
        traces = dict(self.traces)
        selected = signals if signals is not None else sorted(traces)
        rows: list[dict[str, object]] = []
        for name in selected:
            for ts, value in traces.get(name, ()):
                rows.append({"signal": name, "timestamp": ts, "value": value})
        return rows


class Simulator:
    """High-level interactive API for compiling and running designs."""

    def __init__(self, compiled: CompiledModule):
        self._compiled = compiled

    @property
    def module(self) -> Module:
        return self._compiled.module

    @property
    def compiled(self) -> CompiledModule:
        return self._compiled

    @classmethod
    def from_module(cls, module: Module, *, four_state: bool = False) -> "Simulator":
        return cls(compile_module(module, four_state=four_state))

    @classmethod
    def from_sv(cls, source: str, *, top: str | None = None, four_state: bool = False) -> "Simulator":
        from dau_sim.frontends import parse_sv

        module = parse_sv(source, top=top)
        return cls.from_module(module, four_state=four_state)

    @classmethod
    def from_sv_file(cls, path: str, *, top: str | None = None, four_state: bool = False) -> "Simulator":
        from dau_sim.frontends import parse_sv_file

        module = parse_sv_file(path, top=top)
        return cls.from_module(module, four_state=four_state)

    @classmethod
    def from_amaranth(cls, design, *, four_state: bool = False) -> "Simulator":
        from dau_sim.frontends import from_amaranth

        module = from_amaranth(design)
        return cls.from_module(module, four_state=four_state)

    def run(
        self,
        *,
        cycles: int = 10,
        clock_period: timedelta = timedelta(microseconds=1),
        inputs: dict[str, int] | None = None,
        clocks: dict[str, timedelta] | None = None,
    ) -> SimulationResult:
        traces = self._compiled.run(cycles=cycles, clock_period=clock_period, inputs=inputs, clocks=clocks)
        frozen_traces = tuple((name, tuple(values)) for name, values in traces.items())
        return SimulationResult(module_name=self._compiled.module.name, traces=frozen_traces)

    def write_vcd(
        self,
        path: str,
        result: SimulationResult,
        *,
        timescale: str = "1ns",
        signals: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self._compiled.write_vcd(path, dict(result.traces), timescale=timescale, signals=signals, exclude=exclude)
