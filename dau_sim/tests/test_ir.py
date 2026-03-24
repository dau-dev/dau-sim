"""Tests for IR data structures: types, expressions, module construction, validation, and printing."""

import unittest

from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Concat,
    Const,
    EdgePolarity,
    IfElse,
    InitBlock,
    Module,
    Mux,
    Port,
    PortDirection,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
    Slice,
)
from dau_sim.ir.printer import fmt_expr, fmt_module
from dau_sim.ir.stmt import Delay, Finish
from dau_sim.ir.validate import validate_module

# ═══════════════════════════════════════════════════════════════════
# Shape
# ═══════════════════════════════════════════════════════════════════


def test_shape_basics():
    s = Shape(8)
    assert s.width == 8
    assert not s.signed
    assert s.max_value == 255
    assert s.min_value == 0

    s = Shape(8, signed=True)
    assert s.max_value == 127
    assert s.min_value == -128


def test_shape_frozen():
    s = Shape(4)
    try:
        s.width = 5
        raise Exception("Should have raised")
    except AttributeError:
        pass


# ═══════════════════════════════════════════════════════════════════
# Expression IR nodes
# ═══════════════════════════════════════════════════════════════════


def test_const_expr():
    c = Const(shape=Shape(8), value=42)
    assert c.value == 42
    assert c.shape.width == 8


def test_signal_ref():
    r = SignalRef(shape=Shape(1), name="clk")
    assert r.name == "clk"


def test_binary_expr():
    a = Const(shape=Shape(8), value=3)
    b = Const(shape=Shape(8), value=5)
    add = Binary(shape=Shape(8), op=BinaryOp.ADD, left=a, right=b)
    assert add.op == BinaryOp.ADD


def test_mux_expr():
    sel = SignalRef(shape=Shape(1), name="sel")
    a = Const(shape=Shape(8), value=10)
    b = Const(shape=Shape(8), value=20)
    m = Mux(shape=Shape(8), sel=sel, if_true=a, if_false=b)
    assert m.sel == sel


def test_concat_expr():
    hi = Const(shape=Shape(4), value=0xA)
    lo = Const(shape=Shape(4), value=0x5)
    c = Concat(shape=Shape(8), parts=(hi, lo))
    assert c.shape.width == 8


def test_slice_expr():
    val = SignalRef(shape=Shape(8), name="x")
    s = Slice(shape=Shape(4), value=val, low=0, high=4)
    assert s.low == 0
    assert s.high == 4


# ═══════════════════════════════════════════════════════════════════
# Statement / IR node frozen checks
# ═══════════════════════════════════════════════════════════════════


class TestIRNodes(unittest.TestCase):
    def test_delay_frozen(self):
        d = Delay(ticks=5)
        with self.assertRaises(AttributeError):
            d.ticks = 10

    def test_finish_frozen(self):
        f = Finish(exit_code=1)
        with self.assertRaises(AttributeError):
            f.exit_code = 2

    def test_finish_default_exit_code(self):
        f = Finish()
        self.assertEqual(f.exit_code, 0)

    def test_init_block_frozen(self):
        ib = InitBlock(stmts=())
        with self.assertRaises(AttributeError):
            ib.stmts = ()


# ═══════════════════════════════════════════════════════════════════
# Module construction
# ═══════════════════════════════════════════════════════════════════


def _make_counter_module() -> Module:
    """Build a 4-bit counter with synchronous reset.

    Equivalent to:
        module counter(input clk, input rst, output reg [3:0] count);
            always @(posedge clk)
                if (rst) count <= 0;
                else count <= count + 1;
        endmodule
    """
    return Module(
        name="counter",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("rst", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(4)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    IfElse(
                        cond=SignalRef(shape=Shape(1), name="rst"),
                        then_body=(
                            Assign(
                                target="count",
                                value=Const(shape=Shape(4), value=0),
                            ),
                        ),
                        else_body=(
                            Assign(
                                target="count",
                                value=Binary(
                                    shape=Shape(4),
                                    op=BinaryOp.ADD,
                                    left=SignalRef(shape=Shape(4), name="count"),
                                    right=Const(shape=Shape(4), value=1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_module_construction():
    m = _make_counter_module()
    assert m.name == "counter"
    assert len(m.ports) == 3
    assert len(m.seq_blocks) == 1
    assert m.signal_by_name("count").shape == Shape(4)


def test_module_all_signal_names():
    m = _make_counter_module()
    names = m.all_signal_names
    assert names == {"clk", "rst", "count"}


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════


def test_validate_valid_module():
    m = _make_counter_module()
    result = validate_module(m)
    assert result.ok, str(result)


def test_validate_unknown_signal_in_stmt():
    m = Module(
        name="bad",
        ports=(Port(Signal("x", Shape(8)), PortDirection.INPUT),),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="y",  # y doesn't exist
                        value=SignalRef(shape=Shape(8), name="x"),
                    ),
                ),
            ),
        ),
    )
    result = validate_module(m)
    assert not result.ok
    assert "unknown signal" in str(result).lower()


def test_validate_unknown_clock_domain():
    m = Module(
        name="bad",
        ports=(Port(Signal("x", Shape(8)), PortDirection.INPUT),),
        seq_blocks=(
            SeqBlock(
                domain="nonexistent",
                stmts=(
                    Assign(
                        target="x",
                        value=Const(shape=Shape(8), value=0),
                    ),
                ),
            ),
        ),
    )
    result = validate_module(m)
    assert not result.ok
    assert "nonexistent" in str(result)


def test_validate_duplicate_signal():
    m = Module(
        name="dup",
        ports=(Port(Signal("x", Shape(8)), PortDirection.INPUT),),
        signals=(Signal("x", Shape(8)),),  # duplicate
    )
    result = validate_module(m)
    assert not result.ok
    assert "duplicate" in str(result).lower()


# ═══════════════════════════════════════════════════════════════════
# Pretty-printer
# ═══════════════════════════════════════════════════════════════════


def test_fmt_expr_const():
    c = Const(shape=Shape(8), value=42)
    assert "42" in fmt_expr(c)


def test_fmt_expr_binary():
    a = SignalRef(shape=Shape(8), name="a")
    b = SignalRef(shape=Shape(8), name="b")
    add = Binary(shape=Shape(8), op=BinaryOp.ADD, left=a, right=b)
    s = fmt_expr(add)
    assert "+" in s
    assert "a" in s
    assert "b" in s


def test_fmt_module():
    m = _make_counter_module()
    s = fmt_module(m)
    assert "module counter:" in s
    assert "clk" in s
    assert "rst" in s
    assert "count" in s
    assert "seq block" in s
