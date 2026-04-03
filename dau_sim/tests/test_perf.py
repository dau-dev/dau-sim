from dau_sim.benchmarks.compile_partitioning import compile_partitioned_module
from dau_sim.benchmarks.selective_settle import run_partitioned_seq
from dau_sim.ir.expr import Binary, BinaryOp, SignalRef
from dau_sim.ir.module import CombBlock, Module, Port, Signal
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import PortDirection, Shape
from dau_sim.perf import analyze_node_separation, benchmark_module


def _make_two_cluster_module() -> Module:
    a = Signal("a", Shape(8))
    b = Signal("b", Shape(8))
    c = Signal("c", Shape(8))
    d = Signal("d", Shape(8))
    y1 = Signal("y1", Shape(8))
    y2 = Signal("y2", Shape(8))
    y3 = Signal("y3", Shape(8))

    return Module(
        name="clusters",
        ports=(
            Port(a, PortDirection.INPUT),
            Port(b, PortDirection.INPUT),
            Port(c, PortDirection.INPUT),
            Port(d, PortDirection.INPUT),
            Port(y3, PortDirection.OUTPUT),
        ),
        signals=(y1, y2),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        "y1",
                        Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="a"),
                            right=SignalRef(shape=Shape(8), name="b"),
                        ),
                    ),
                )
            ),
            CombBlock(
                stmts=(
                    Assign(
                        "y2",
                        Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="c"),
                            right=SignalRef(shape=Shape(8), name="d"),
                        ),
                    ),
                )
            ),
            CombBlock(
                stmts=(
                    Assign(
                        "y3",
                        Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="y1"),
                            right=SignalRef(shape=Shape(8), name="y2"),
                        ),
                    ),
                )
            ),
        ),
    )


def test_analyze_node_separation_reports_components() -> None:
    stats = analyze_node_separation(_make_two_cluster_module())

    assert stats.comb_blocks == 3
    assert stats.connected_components >= 1
    assert stats.largest_component >= 1


def test_benchmark_module_returns_positive_metrics() -> None:
    result = benchmark_module(
        _make_two_cluster_module(),
        cycles=10,
        repeats=1,
        warmup=0,
        inputs={"a": 1, "b": 2, "c": 3, "d": 4},
    )

    assert result.compile_seconds_median >= 0
    assert result.run_seconds_median >= 0
    assert result.cycles_per_second > 0


def test_benchmark_compile_partitioning_smoke() -> None:
    compile_partitioned_module(16)


def test_benchmark_selective_settle_smoke() -> None:
    run_partitioned_seq(16, 8, cycles=20)
