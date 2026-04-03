from __future__ import annotations

from dau_sim.compiler import compile_module
from dau_sim.ir.expr import Binary, BinaryOp, Const, SignalRef
from dau_sim.ir.module import ClockDomain, CombBlock, Module, SeqBlock, Signal
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import Shape


def _make_partitioned_seq_module(n: int, stmts_per_component: int) -> Module:
    source_signals = tuple(Signal(f"s{i}", Shape(8)) for i in range(n))
    derived_signals = tuple(Signal(f"o{i}_{j}", Shape(8)) for i in range(n) for j in range(stmts_per_component))

    comb_blocks = []
    for i in range(n):
        stmts = []
        for j in range(stmts_per_component):
            stmts.append(
                Assign(
                    target=f"o{i}_{j}",
                    value=Binary(
                        shape=Shape(8),
                        op=BinaryOp.ADD,
                        left=SignalRef(shape=Shape(8), name=f"s{i}"),
                        right=Const(shape=Shape(8), value=j & 0xFF),
                    ),
                )
            )
        comb_blocks.append(CombBlock(stmts=tuple(stmts)))

    seq_blocks = (
        SeqBlock(
            "sync",
            stmts=(
                Assign(
                    target="s0",
                    value=Binary(
                        shape=Shape(8),
                        op=BinaryOp.ADD,
                        left=SignalRef(shape=Shape(8), name="s0"),
                        right=Const(shape=Shape(8), value=1),
                    ),
                ),
            ),
        ),
    )

    return Module(
        name=f"selective_settle_{n}_{stmts_per_component}",
        signals=source_signals + derived_signals,
        comb_blocks=tuple(comb_blocks),
        seq_blocks=seq_blocks,
        clock_domains=(ClockDomain("sync", clk="clk"),),
    )


def run_partitioned_seq(n: int, stmts_per_component: int, cycles: int = 200) -> None:
    """Compile then run a partitionable sequential design."""
    compiled = compile_module(_make_partitioned_seq_module(n, stmts_per_component))
    compiled.run(cycles=cycles)
