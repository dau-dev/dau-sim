"""Tests for non-synthesizable constructs.

Covers: initial blocks, $display (Print), assertions, $finish (Finish),
delay statements, and SimulationFinish exception.
"""

import io
import unittest
from contextlib import redirect_stdout

from dau_sim.compiler.compile import SimulationFinish, compile_module
from dau_sim.ir import (
    Assert,
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    IfElse,
    InitBlock,
    Module,
    Port,
    PortDirection,
    Print,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
)
from dau_sim.ir.stmt import Delay, Finish


def _make_signal(name, width=8, init=0):
    return Signal(name, Shape(width), init=init)


def _make_port(name, width=8, direction=PortDirection.INPUT, init=0):
    return Port(Signal(name, Shape(width), init=init), direction)


class TestInitBlock(unittest.TestCase):
    def test_init_block_sets_value(self):
        """Initial block assigns to signal before simulation starts."""
        a = _make_signal("a", 8, init=0)
        y = _make_signal("y", 8, init=0)
        mod = Module(
            name="init_test",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", SignalRef(Shape(8), "a")),)),),
            init_blocks=(InitBlock(stmts=(Assign("a", Const(Shape(8), 42)),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        # Init block set a=42, comb block copies a→y
        self.assertEqual(traces["a"][0][1], 42)
        self.assertEqual(traces["y"][0][1], 42)

    def test_init_block_multiple_assigns(self):
        """Multiple assignments in an init block."""
        a = _make_signal("a", 8, init=0)
        b = _make_signal("b", 8, init=0)
        y = _make_signal("y", 8, init=0)
        mod = Module(
            name="multi_init",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(b, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
            init_blocks=(
                InitBlock(
                    stmts=(
                        Assign("a", Const(Shape(8), 10)),
                        Assign("b", Const(Shape(8), 20)),
                    )
                ),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        self.assertEqual(traces["y"][0][1], 30)

    def test_init_block_overrides_port_init(self):
        """Init block values take priority over port init values."""
        a = _make_signal("a", 8, init=5)
        mod = Module(
            name="override_init",
            ports=(Port(a, PortDirection.INPUT),),
            init_blocks=(InitBlock(stmts=(Assign("a", Const(Shape(8), 99)),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        # Init block overrides the port's init=5
        self.assertEqual(traces["a"][0][1], 99)

    def test_multiple_init_blocks(self):
        """Multiple init blocks execute in order."""
        a = _make_signal("a", 8)
        mod = Module(
            name="multi_block",
            ports=(Port(a, PortDirection.OUTPUT),),
            init_blocks=(
                InitBlock(stmts=(Assign("a", Const(Shape(8), 10)),)),
                InitBlock(stmts=(Assign("a", Const(Shape(8), 20)),)),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        # Second init block wins (last assignment)
        self.assertEqual(traces["a"][0][1], 20)

    def test_init_block_with_sequential(self):
        """Init blocks work with sequential modules."""
        clk = _make_signal("clk", 1)
        count = _make_signal("count", 8, init=0)
        mod = Module(
            name="init_seq",
            ports=(
                Port(clk, PortDirection.INPUT),
                Port(count, PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", "clk"),),
            seq_blocks=(
                SeqBlock(
                    domain="sync",
                    stmts=(Assign("count", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "count"), Const(Shape(8), 1))),),
                ),
            ),
            init_blocks=(InitBlock(stmts=(Assign("count", Const(Shape(8), 100)),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=3)
        # Count starts at 100 (from init block), then increments
        vals = [v for _, v in traces["count"]]
        self.assertEqual(vals[0], 101)  # first posedge: 100+1
        self.assertEqual(vals[1], 102)
        self.assertEqual(vals[2], 103)


class TestPrintStmt(unittest.TestCase):
    def test_display_basic(self):
        """Print stmt outputs to stdout during simulation."""
        a = _make_signal("a", 8, init=42)
        mod = Module(
            name="display_test",
            ports=(Port(a, PortDirection.INPUT),),
            comb_blocks=(CombBlock(stmts=(Print("a = {}", (SignalRef(Shape(8), "a"),)),)),),
        )
        cm = compile_module(mod)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cm.run(cycles=1)
        self.assertIn("a = 42", buf.getvalue())

    def test_display_format_multiple_args(self):
        """Print with multiple arguments."""
        a = _make_signal("a", 8, init=10)
        b = _make_signal("b", 8, init=20)
        mod = Module(
            name="display_multi",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(b, PortDirection.INPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Print("a={} b={}", (SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
        )
        cm = compile_module(mod)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cm.run(cycles=1)
        self.assertIn("a=10 b=20", buf.getvalue())

    def test_display_in_init_block(self):
        """Print in init block executes at time 0."""
        a = _make_signal("a", 8, init=7)
        mod = Module(
            name="display_init",
            ports=(Port(a, PortDirection.INPUT),),
            init_blocks=(InitBlock(stmts=(Print("init: a={}", (SignalRef(Shape(8), "a"),)),)),),
        )
        cm = compile_module(mod)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cm.run(cycles=1)
        self.assertIn("init: a=7", buf.getvalue())


class TestAssertStmt(unittest.TestCase):
    def test_assert_pass(self):
        """Assertion passes when condition is true."""
        a = _make_signal("a", 8, init=1)
        mod = Module(
            name="assert_pass",
            ports=(Port(a, PortDirection.INPUT),),
            comb_blocks=(CombBlock(stmts=(Assert(cond=SignalRef(Shape(8), "a"), message="a should be nonzero"),)),),
        )
        cm = compile_module(mod)
        # Should not raise
        cm.run(cycles=1)

    def test_assert_fail(self):
        """Assertion raises when condition is false."""
        a = _make_signal("a", 8, init=0)
        mod = Module(
            name="assert_fail",
            ports=(Port(a, PortDirection.INPUT),),
            comb_blocks=(CombBlock(stmts=(Assert(cond=SignalRef(Shape(8), "a"), message="a must be nonzero"),)),),
        )
        cm = compile_module(mod)
        with self.assertRaises(AssertionError) as ctx:
            cm.run(cycles=1)
        self.assertIn("a must be nonzero", str(ctx.exception))

    def test_assert_default_message(self):
        """Assert with no message uses default."""
        a = _make_signal("a", 8, init=0)
        mod = Module(
            name="assert_default",
            ports=(Port(a, PortDirection.INPUT),),
            comb_blocks=(CombBlock(stmts=(Assert(cond=SignalRef(Shape(8), "a")),)),),
        )
        cm = compile_module(mod)
        with self.assertRaises(AssertionError) as ctx:
            cm.run(cycles=1)
        self.assertIn("assertion failed", str(ctx.exception))

    def test_assert_in_init_block(self):
        """Assert in init block fires at time 0."""
        a = _make_signal("a", 8, init=0)
        mod = Module(
            name="assert_init",
            ports=(Port(a, PortDirection.INPUT),),
            init_blocks=(InitBlock(stmts=(Assert(cond=SignalRef(Shape(8), "a"), message="init assert failed"),)),),
        )
        cm = compile_module(mod)
        with self.assertRaises(AssertionError) as ctx:
            cm.run(cycles=1)
        self.assertIn("init assert failed", str(ctx.exception))


class TestFinishStmt(unittest.TestCase):
    def test_finish_in_init_block(self):
        """$finish in init block returns single-point traces."""
        a = _make_signal("a", 8, init=42)
        mod = Module(
            name="finish_init",
            ports=(Port(a, PortDirection.INPUT),),
            init_blocks=(InitBlock(stmts=(Finish(exit_code=0),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=10)
        # Should have exactly one data point (no simulation ran)
        self.assertEqual(len(traces["a"]), 1)
        self.assertEqual(traces["a"][0][1], 42)

    def test_finish_in_comb_block(self):
        """$finish in comb block stops simulation after first eval."""
        a = _make_signal("a", 8, init=5)
        mod = Module(
            name="finish_comb",
            ports=(Port(a, PortDirection.INPUT),),
            comb_blocks=(CombBlock(stmts=(Finish(exit_code=1),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=10)
        # Should stop after first evaluation
        self.assertLessEqual(len(traces["a"]), 2)

    def test_finish_conditional(self):
        """$finish only fires when count reaches threshold."""
        clk = _make_signal("clk", 1)
        count = _make_signal("count", 8, init=0)
        mod = Module(
            name="finish_cond",
            ports=(
                Port(clk, PortDirection.INPUT),
                Port(count, PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", "clk"),),
            seq_blocks=(
                SeqBlock(
                    domain="sync",
                    stmts=(
                        Assign("count", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "count"), Const(Shape(8), 1))),
                        IfElse(
                            cond=Binary(Shape(1), BinaryOp.EQ, SignalRef(Shape(8), "count"), Const(Shape(8), 3)),
                            then_body=(Finish(),),
                        ),
                    ),
                ),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=20)
        # Counter should stop when it hits 3
        vals = [v for _, v in traces["count"]]
        last_val = vals[-1]
        # After count reaches 3, finish fires, stopping further increments
        self.assertLessEqual(last_val, 4)

    def test_simulation_finish_exception(self):
        """SimulationFinish stores exit code."""
        exc = SimulationFinish(42)
        self.assertEqual(exc.exit_code, 42)
        self.assertIn("42", str(exc))


class TestDelayStmt(unittest.TestCase):
    def test_delay_ir_node(self):
        """Delay IR node stores ticks."""
        d = Delay(ticks=10)
        self.assertEqual(d.ticks, 10)

    def test_delay_in_init_block(self):
        """Delay in init block is no-op (doesn't crash)."""
        a = _make_signal("a", 8, init=0)
        mod = Module(
            name="delay_test",
            ports=(Port(a, PortDirection.INPUT),),
            init_blocks=(
                InitBlock(
                    stmts=(
                        Delay(ticks=5),
                        Assign("a", Const(Shape(8), 99)),
                    )
                ),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        # The delay is currently a no-op; the assign still executes
        self.assertEqual(traces["a"][0][1], 99)

    def test_delay_in_comb_block(self):
        """Delay in comb block is ignored."""
        a = _make_signal("a", 8, init=0)
        y = _make_signal("y", 8, init=0)
        mod = Module(
            name="delay_comb",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            comb_blocks=(
                CombBlock(
                    stmts=(
                        Delay(ticks=10),
                        Assign("y", SignalRef(Shape(8), "a")),
                    )
                ),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1)
        self.assertEqual(traces["y"][0][1], 0)


class TestIntegration(unittest.TestCase):
    def test_init_print_assert(self):
        """Full flow: init sets values, print displays them, assert checks."""
        a = _make_signal("a", 8, init=0)
        b = _make_signal("b", 8, init=0)
        y = _make_signal("y", 8, init=0)
        mod = Module(
            name="integration",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(b, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            comb_blocks=(
                CombBlock(
                    stmts=(
                        Assign("y", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),
                        Assert(cond=Binary(Shape(1), BinaryOp.EQ, SignalRef(Shape(8), "y"), Const(Shape(8), 30)), message="y should be 30"),
                        Print("Result: y={}", (SignalRef(Shape(8), "y"),)),
                    )
                ),
            ),
            init_blocks=(
                InitBlock(
                    stmts=(
                        Assign("a", Const(Shape(8), 10)),
                        Assign("b", Const(Shape(8), 20)),
                    )
                ),
            ),
        )
        cm = compile_module(mod)
        buf = io.StringIO()
        with redirect_stdout(buf):
            traces = cm.run(cycles=1)
        self.assertEqual(traces["y"][0][1], 30)
        self.assertIn("Result: y=30", buf.getvalue())

    def test_sequential_init_block(self):
        """Init block with a sequential counter: count starts at custom value."""
        clk = _make_signal("clk", 1)
        en = _make_signal("en", 1, init=1)
        count = _make_signal("count", 8, init=0)
        mod = Module(
            name="seq_init_counter",
            ports=(
                Port(clk, PortDirection.INPUT),
                Port(en, PortDirection.INPUT),
                Port(count, PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", "clk"),),
            seq_blocks=(
                SeqBlock(
                    domain="sync",
                    stmts=(
                        IfElse(
                            cond=SignalRef(Shape(1), "en"),
                            then_body=(Assign("count", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "count"), Const(Shape(8), 1))),),
                        ),
                    ),
                ),
            ),
            init_blocks=(InitBlock(stmts=(Assign("count", Const(Shape(8), 50)),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=5)
        vals = [v for _, v in traces["count"]]
        # Starts at 50 (init block), en=1, so increments each cycle
        self.assertEqual(vals[0], 51)
        self.assertEqual(vals[1], 52)
        self.assertEqual(vals[2], 53)


if __name__ == "__main__":
    unittest.main()
