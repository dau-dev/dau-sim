"""The generated run loop reproduces the CSP engine's tick semantics on the
cases where the two are easy to get apart: a component rescheduled by a signal
it also writes, a change made and reverted within one tick, and ``$finish``
inside a tick and during the initial combinational seed."""

import io
from contextlib import redirect_stdout

import pytest

from dau_sim.compiler import compile_module
from dau_sim.compiler.compile import SimulationFinish
from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    EdgePolarity,
    IfElse,
    Module,
    Port,
    PortDirection,
    Print,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
)
from dau_sim.ir.stmt import Finish

W = Shape(8)


def _in(name, init=0, width=W):
    return Port(signal=Signal(name=name, shape=width, init=init), direction=PortDirection.INPUT)


def _out(name, width=W):
    return Port(signal=Signal(name=name, shape=width, init=0), direction=PortDirection.OUTPUT)


def _ref(name, width=W):
    return SignalRef(shape=width, name=name)


def _add(a, b):
    return Binary(shape=W, op=BinaryOp.ADD, left=a, right=b)


def test_a_component_is_rescheduled_by_a_signal_it_also_writes():
    """The engine routes components by every signal they touch, reads and
    writes alike: a sequential write to a signal the component also writes
    re-evaluates it."""
    m = Module(
        name="shared",
        ports=(_in("clk", width=Shape(1)), _in("one", init=1), _out("acc")),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(SeqBlock(domain="sync", stmts=(Assign(target="acc", value=Const(shape=W, value=5)),)),),
        comb_blocks=(CombBlock(stmts=(Assign(target="acc", value=_add(_ref("acc"), _ref("one"))),)),),
    )
    traces = compile_module(m).run(cycles=2, inputs={"one": 1})
    assert [v for _, v in traces["acc"]] == [6, 6]


def test_a_change_reverted_within_one_tick_still_settles_the_component():
    """Two domains on one clock set q to 1 and back to 0 in the same tick; the
    engine's changed set keeps the first write, so the component runs."""
    m = Module(
        name="revert",
        ports=(_in("clk", width=Shape(1)), _out("q", width=Shape(1)), _out("r", width=Shape(1))),
        clock_domains=(
            ClockDomain(name="a", clk="clk", edge=EdgePolarity.POSEDGE),
            ClockDomain(name="b", clk="clk", edge=EdgePolarity.POSEDGE),
        ),
        seq_blocks=(
            SeqBlock(domain="a", stmts=(Assign(target="q", value=Const(shape=Shape(1), value=1)),)),
            SeqBlock(domain="b", stmts=(Assign(target="q", value=Const(shape=Shape(1), value=0)),)),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Print(format_str="q={}", args=(_ref("q", Shape(1)),)),
                    Assign(target="r", value=_ref("q", Shape(1))),
                )
            ),
        ),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        traces = compile_module(m).run(cycles=3)
    assert [v for _, v in traces["q"]] == [0, 0, 0]
    assert buf.getvalue().splitlines() == ["q=0"] * 4  # the seed, then once per edge


def test_finish_inside_a_tick_keeps_that_ticks_writes_and_its_trace():
    """A block's writes before ``$finish`` take effect and the finishing tick
    is recorded, as a simulator finishing at the end of the time step does.
    (The CSP engine discards the raising block's writes; that is the one
    place the loop deliberately differs.)"""
    m = Module(
        name="finishing",
        ports=(_in("clk", width=Shape(1)), _out("count")),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    Assign(target="count", value=_add(_ref("count"), Const(shape=W, value=1))),
                    IfElse(
                        cond=Binary(shape=Shape(1), op=BinaryOp.EQ, left=_ref("count"), right=Const(shape=W, value=3)),
                        then_body=(Finish(),),
                        else_body=(),
                    ),
                ),
            ),
        ),
    )
    traces = compile_module(m).run(cycles=10)
    assert [v for _, v in traces["count"]] == [1, 2, 3]


def test_finish_during_the_combinational_seed_propagates():
    m = Module(
        name="seed_finish",
        ports=(_in("clk", width=Shape(1)), _in("a", width=Shape(1)), _out("q", width=Shape(1))),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(SeqBlock(domain="sync", stmts=(Assign(target="q", value=_ref("a", Shape(1))),)),),
        comb_blocks=(CombBlock(stmts=(IfElse(cond=_ref("a", Shape(1)), then_body=(Finish(),), else_body=()),)),),
    )
    with pytest.raises(SimulationFinish):
        compile_module(m).run(cycles=2, inputs={"a": 1})
