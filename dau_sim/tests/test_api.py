import pytest

from dau_sim.api import Simulator
from dau_sim.ir.expr import Binary, BinaryOp, SignalRef
from dau_sim.ir.module import CombBlock, Module, Port, Signal
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import PortDirection, Shape


def _make_adder_module() -> Module:
    a = Signal(name="a", shape=Shape(8))
    b = Signal(name="b", shape=Shape(8))
    y = Signal(name="y", shape=Shape(8))
    return Module(
        name="adder",
        ports=(
            Port(signal=a, direction=PortDirection.INPUT),
            Port(signal=b, direction=PortDirection.INPUT),
            Port(signal=y, direction=PortDirection.OUTPUT),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="y",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="a"),
                            right=SignalRef(shape=Shape(8), name="b"),
                        ),
                    ),
                )
            ),
        ),
    )


def test_simulator_from_module_run_and_latest() -> None:
    sim = Simulator.from_module(_make_adder_module())
    result = sim.run(cycles=1, inputs={"a": 10, "b": 22})

    assert result.module_name == "adder"
    assert result.latest("y") == 32


def test_simulation_result_to_rows() -> None:
    sim = Simulator.from_module(_make_adder_module())
    result = sim.run(cycles=1, inputs={"a": 5, "b": 9})
    rows = result.to_rows(signals=["y"])

    assert rows
    assert {"signal", "timestamp", "value"}.issubset(rows[0].keys())
    assert rows[-1]["signal"] == "y"
    assert rows[-1]["value"] == 14


def test_the_package_top_level_loads_the_simulator_on_first_use() -> None:
    """Importing dau_sim (as hydra does to reach dau_sim.config) does not
    import csp; the simulator's names resolve when asked for."""
    import subprocess
    import sys

    probe = "import sys, dau_sim; assert 'csp' not in sys.modules, 'csp loaded at import'"
    subprocess.run([sys.executable, "-c", probe], check=True)

    import dau_sim
    from dau_sim import api

    assert dau_sim.Simulator is api.Simulator
    assert dau_sim.SimulationResult is api.SimulationResult
    with pytest.raises(AttributeError, match="no attribute 'nothing_here'"):
        dau_sim.nothing_here  # noqa: B018
