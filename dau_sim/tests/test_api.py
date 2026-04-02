from dau_sim.api import Simulator
from dau_sim.ir.expr import Binary, BinaryOp, SignalRef
from dau_sim.ir.module import CombBlock, Module, Port, Signal
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import PortDirection, Shape


def _make_adder_module() -> Module:
    a = Signal("a", Shape(8))
    b = Signal("b", Shape(8))
    y = Signal("y", Shape(8))
    return Module(
        name="adder",
        ports=(
            Port(a, PortDirection.INPUT),
            Port(b, PortDirection.INPUT),
            Port(y, PortDirection.OUTPUT),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        "y",
                        Binary(
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
