import unittest

from amaranth import Module as AModule
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from dau_sim.compiler import compile_module
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
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
)
from dau_sim.testbench import TestbenchContext, TestbenchTimeout


def _make_counter_module(width=8, name="counter"):
    """8-bit counter with 'en' input, 'count' output, sync domain."""
    en = Signal("en", Shape(1))
    count = Signal("count", Shape(width))
    clk = Signal("clk", Shape(1))
    rst = Signal("rst", Shape(1))
    return Module(
        name=name,
        ports=(
            Port(en, PortDirection.INPUT),
            Port(count, PortDirection.OUTPUT),
            Port(clk, PortDirection.INPUT),
            Port(rst, PortDirection.INPUT),
        ),
        signals=(),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
        comb_blocks=(),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    IfElse(
                        cond=SignalRef(Shape(1), "en"),
                        then_body=(
                            Assign(
                                "count",
                                Binary(
                                    Shape(width),
                                    BinaryOp.ADD,
                                    SignalRef(Shape(width), "count"),
                                    Const(Shape(width), 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_adder_module():
    """Combinational 8-bit adder: y = a + b."""
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
        signals=(),
        clock_domains=(),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        "y",
                        Binary(
                            Shape(8),
                            BinaryOp.ADD,
                            SignalRef(Shape(8), "a"),
                            SignalRef(Shape(8), "b"),
                        ),
                    ),
                )
            ),
        ),
        seq_blocks=(),
    )


def _make_updown_counter():
    """Counter with en, dir (0=up, 1=down), 8-bit count."""
    en = Signal("en", Shape(1))
    dir_ = Signal("dir", Shape(1))
    count = Signal("count", Shape(8))
    clk = Signal("clk", Shape(1))
    rst = Signal("rst", Shape(1))
    return Module(
        name="updown",
        ports=(
            Port(en, PortDirection.INPUT),
            Port(dir_, PortDirection.INPUT),
            Port(count, PortDirection.OUTPUT),
            Port(clk, PortDirection.INPUT),
            Port(rst, PortDirection.INPUT),
        ),
        signals=(),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
        comb_blocks=(),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    IfElse(
                        cond=SignalRef(Shape(1), "en"),
                        then_body=(
                            IfElse(
                                cond=SignalRef(Shape(1), "dir"),
                                then_body=(
                                    Assign(
                                        "count",
                                        Binary(
                                            Shape(8),
                                            BinaryOp.SUB,
                                            SignalRef(Shape(8), "count"),
                                            Const(Shape(8), 1),
                                        ),
                                    ),
                                ),
                                else_body=(
                                    Assign(
                                        "count",
                                        Binary(
                                            Shape(8),
                                            BinaryOp.ADD,
                                            SignalRef(Shape(8), "count"),
                                            Const(Shape(8), 1),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestContextInit(unittest.TestCase):
    def test_initial_signals(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        self.assertEqual(ctx.get("en"), 0)
        self.assertEqual(ctx.get("count"), 0)
        self.assertEqual(ctx.get("clk"), 0)
        self.assertEqual(ctx.get("rst"), 0)

    def test_initial_cycle_zero(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        self.assertEqual(ctx.cycle, 0)

    def test_unknown_signal_get(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        with self.assertRaises(KeyError):
            ctx.get("nonexistent")

    def test_unknown_signal_set(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        with self.assertRaises(KeyError):
            ctx.set("nonexistent", 1)

    def test_history_has_initial(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        # cycle-0 snapshot recorded
        self.assertEqual(ctx.history_of("count"), [0])


class TestSetGetTick(unittest.TestCase):
    def test_set_and_get(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        self.assertEqual(ctx.get("en"), 1)

    def test_tick_advances_cycle(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick()
        self.assertEqual(ctx.cycle, 1)

    def test_tick_n_advances_n(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        self.assertEqual(ctx.cycle, 5)

    def test_tick_zero_raises(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        with self.assertRaises(ValueError):
            ctx.tick(0)

    def test_tick_negative_raises(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        with self.assertRaises(ValueError):
            ctx.tick(-1)


class TestSequentialCounter(unittest.TestCase):
    def test_counter_counts_when_enabled(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        self.assertEqual(ctx.get("count"), 5)

    def test_counter_holds_when_disabled(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        ctx.set("en", 0)
        ctx.tick(5)
        self.assertEqual(ctx.get("count"), 5)

    def test_counter_resume(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(3)
        ctx.set("en", 0)
        ctx.tick(2)
        ctx.set("en", 1)
        ctx.tick(4)
        # 3 + 4 = 7
        self.assertEqual(ctx.get("count"), 7)

    def test_counter_single_step(self):
        """Tick one cycle at a time, verify each step."""
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        for expected in range(1, 11):
            ctx.tick()
            self.assertEqual(ctx.get("count"), expected)

    def test_counter_batch_vs_single(self):
        """Batch tick(10) should give same result as 10 × tick(1)."""
        cm = compile_module(_make_counter_module())
        ctx_batch = TestbenchContext(cm)
        ctx_batch.set("en", 1)
        ctx_batch.tick(10)

        ctx_single = TestbenchContext(cm)
        ctx_single.set("en", 1)
        for _ in range(10):
            ctx_single.tick()

        self.assertEqual(ctx_batch.get("count"), ctx_single.get("count"))
        self.assertEqual(ctx_batch.get("count"), 10)


class TestCombTestbench(unittest.TestCase):
    def test_adder_basic(self):
        cm = compile_module(_make_adder_module())
        ctx = TestbenchContext(cm)
        ctx.set("a", 10)
        ctx.set("b", 20)
        ctx.tick()
        self.assertEqual(ctx.get("y"), 30)

    def test_adder_change_inputs(self):
        cm = compile_module(_make_adder_module())
        ctx = TestbenchContext(cm)
        ctx.set("a", 100)
        ctx.set("b", 55)
        ctx.tick()
        self.assertEqual(ctx.get("y"), 155)
        ctx.set("a", 200)
        ctx.set("b", 50)
        ctx.tick()
        self.assertEqual(ctx.get("y"), 250)


class TestAssertions(unittest.TestCase):
    def test_assert_eq_pass(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        ctx.assert_eq("count", 5)  # Should not raise

    def test_assert_eq_fail(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        with self.assertRaises(AssertionError) as cm_ctx:
            ctx.assert_eq("count", 99)
        self.assertIn("count", str(cm_ctx.exception))
        self.assertIn("99", str(cm_ctx.exception))

    def test_assert_eq_message(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.tick(1)
        with self.assertRaises(AssertionError) as cm_ctx:
            ctx.assert_eq("count", 42, msg="custom message")
        self.assertIn("custom message", str(cm_ctx.exception))

    def test_assert_neq_pass(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        ctx.assert_neq("count", 0)  # Should not raise

    def test_assert_neq_fail(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        # count is 0, so assert_neq(0) should fail
        with self.assertRaises(AssertionError):
            ctx.assert_neq("count", 0)

    def test_assert_stable_pass(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        # en=0, count should stay at 0
        ctx.tick(5)
        ctx.assert_stable("count")

    def test_assert_stable_fail(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(3)
        with self.assertRaises(AssertionError):
            ctx.assert_stable("count")

    def test_assert_stable_with_cycles(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        ctx.set("en", 0)
        ctx.tick(3)
        # Last 3 cycles: count stable at 5
        ctx.assert_stable("count", cycles=3)
        # But full history is not stable
        with self.assertRaises(AssertionError):
            ctx.assert_stable("count")

    def test_assert_changed_pass(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(3)
        ctx.assert_changed("count")

    def test_assert_changed_fail(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.tick(3)
        with self.assertRaises(AssertionError):
            ctx.assert_changed("count")


class TestHistory(unittest.TestCase):
    def test_history_counter(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.tick(5)
        h = ctx.history_of("count")
        # cycle 0: init=0, then cycles 1-5: 1,2,3,4,5
        self.assertEqual(h[0], 0)  # initial
        self.assertEqual(h[-1], 5)
        self.assertEqual(len(h), 6)  # 1 initial + 5 ticks

    def test_history_unknown_signal(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm)
        with self.assertRaises(KeyError):
            ctx.history_of("bogus")


class TestTimeout(unittest.TestCase):
    def test_timeout_raises(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm, max_cycles=10)
        ctx.set("en", 1)
        with self.assertRaises(TestbenchTimeout):
            ctx.tick(20)

    def test_timeout_at_boundary(self):
        cm = compile_module(_make_counter_module())
        ctx = TestbenchContext(cm, max_cycles=10)
        ctx.tick(5)
        ctx.tick(5)  # exactly 10 — should work
        with self.assertRaises(TestbenchTimeout):
            ctx.tick(1)  # 11 — over limit


class TestRunTestbench(unittest.TestCase):
    def test_run_testbench_pass(self):
        cm = compile_module(_make_counter_module())

        def my_test(ctx):
            ctx.set("en", 1)
            ctx.tick(5)
            ctx.assert_eq("count", 5)

        result = cm.run_testbench(my_test)
        self.assertTrue(result.passed)
        self.assertEqual(result.cycle, 5)
        self.assertEqual(result.signals["count"], 5)

    def test_run_testbench_fail(self):
        cm = compile_module(_make_counter_module())

        def my_test(ctx):
            ctx.set("en", 1)
            ctx.tick(5)
            ctx.assert_eq("count", 99)

        with self.assertRaises(AssertionError):
            cm.run_testbench(my_test)

    def test_run_testbench_returns_history(self):
        cm = compile_module(_make_counter_module())

        def my_test(ctx):
            ctx.set("en", 1)
            ctx.tick(3)

        result = cm.run_testbench(my_test)
        self.assertIn("count", result.history)
        self.assertEqual(len(result.history["count"]), 4)  # init + 3


class TestUpDownCounter(unittest.TestCase):
    def test_count_up(self):
        cm = compile_module(_make_updown_counter())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.set("dir", 0)  # up
        ctx.tick(5)
        ctx.assert_eq("count", 5)

    def test_count_down(self):
        cm = compile_module(_make_updown_counter())
        ctx = TestbenchContext(cm)
        # First count up
        ctx.set("en", 1)
        ctx.set("dir", 0)
        ctx.tick(10)
        ctx.assert_eq("count", 10)
        # Now count down
        ctx.set("dir", 1)
        ctx.tick(3)
        ctx.assert_eq("count", 7)

    def test_direction_change(self):
        cm = compile_module(_make_updown_counter())
        ctx = TestbenchContext(cm)
        ctx.set("en", 1)
        ctx.set("dir", 0)
        ctx.tick(5)
        self.assertEqual(ctx.get("count"), 5)
        ctx.set("dir", 1)
        ctx.tick(2)
        self.assertEqual(ctx.get("count"), 3)
        ctx.set("dir", 0)
        ctx.tick(3)
        self.assertEqual(ctx.get("count"), 6)


class Phase5Counter(wiring.Component):
    en: In(1)
    count: Out(8)

    def elaborate(self, platform):
        m = AModule()
        with m.If(self.en):
            m.d.sync += self.count.eq(self.count + 1)
        return m


class Phase5Mux(wiring.Component):
    sel: In(1)
    a: In(8)
    b: In(8)
    y: Out(8)

    def elaborate(self, platform):
        m = AModule()
        with m.If(self.sel):
            m.d.comb += self.y.eq(self.b)
        with m.Else():
            m.d.comb += self.y.eq(self.a)
        return m


class TestAmaranthTestbench(unittest.TestCase):
    def test_amaranth_counter(self):
        """Full Amaranth → IR → compile → testbench pipeline."""
        from dau_sim.frontends import from_amaranth

        ir = from_amaranth(Phase5Counter())
        cm = compile_module(ir)

        def test_fn(ctx):
            ctx.set("en", 1)
            for i in range(1, 11):
                ctx.tick()
                ctx.assert_eq("count", i)
            ctx.set("en", 0)
            ctx.tick(5)
            ctx.assert_eq("count", 10)

        result = cm.run_testbench(test_fn)
        self.assertTrue(result.passed)
        self.assertEqual(result.cycle, 15)

    def test_amaranth_mux(self):
        """Amaranth combinational design testbench."""
        from dau_sim.frontends import from_amaranth

        ir = from_amaranth(Phase5Mux())
        cm = compile_module(ir)

        def test_fn(ctx):
            ctx.set("a", 42)
            ctx.set("b", 99)
            ctx.set("sel", 0)
            ctx.tick()
            ctx.assert_eq("y", 42)
            ctx.set("sel", 1)
            ctx.tick()
            ctx.assert_eq("y", 99)

        result = cm.run_testbench(test_fn)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
