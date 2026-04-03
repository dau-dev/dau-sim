from __future__ import annotations

from dau_sim.compiler import compile_module
from dau_sim.ir.expr import Binary, BinaryOp, Const, SignalRef
from dau_sim.ir.module import ClockDomain, CombBlock, Module, Port, Signal
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import PortDirection, Shape


def _make_partitioned_comb_module(n: int) -> Module:
    a = Signal("a", Shape(8))
    internal_signals = tuple(Signal(f"s{i}", Shape(8)) for i in range(n))
    output_signals = tuple(Signal(f"o{i}", Shape(8)) for i in range(n))

    comb_blocks = tuple(
        CombBlock(
            stmts=(
                Assign(
                    target=f"o{i}",
                    value=Binary(
                        shape=Shape(8),
                        op=BinaryOp.ADD,
                        left=SignalRef(shape=Shape(8), name="a" if i == 0 else f"s{i}"),
                        right=Const(shape=Shape(8), value=i & 0xFF),
                    ),
                ),
            )
        )
        for i in range(n)
    )

    return Module(
        name=f"partition_compile_{n}",
        ports=(Port(a, PortDirection.INPUT),),
        signals=internal_signals + output_signals,
        comb_blocks=comb_blocks,
        clock_domains=(ClockDomain("sync", clk="clk"),),
    )


def compile_partitioned_module(n: int) -> None:
    """Compile a synthetic module with many disconnected comb blocks."""
    compile_module(_make_partitioned_comb_module(n))
