"""Tests for sequential simulation: clock domains, registers, resets.

Circuits tested:
- Counter with synchronous reset (basic end-to-end)
- D flip-flop
- Shift register (4-bit serial-in/parallel-out)
- Counter with sync/async reset
- Negedge-sensitive counter
- Counter with combinational output decode (mixed seq+comb)
- Dual-clock domain
- Four-state sequential simulation
"""

from datetime import timedelta

from dau_sim.compiler import compile_module
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
    Module,
    Port,
    PortDirection,
    ResetStyle,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
    Slice,
)


def _make_counter_module() -> Module:
    """4-bit counter with synchronous reset (if/else in seq block)."""
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


def test_compile_and_run_counter():
    """End-to-end: build a 4-bit counter IR, compile to CSP, simulate 10 cycles."""
    m = _make_counter_module()
    compiled = compile_module(m)
    traces = compiled.run(cycles=10, inputs={"clk": 0, "rst": 0})

    assert "count" in traces
    count_values = [v for _, v in traces["count"]]

    assert len(count_values) == 10
    for i, v in enumerate(count_values):
        expected = (i + 1) & 0xF  # 4-bit wrapping
        assert v == expected, f"cycle {i}: expected {expected}, got {v}"


def test_compile_and_run_counter_with_reset():
    """Counter with reset held high should stay at 0."""
    m = _make_counter_module()
    compiled = compile_module(m)
    traces = compiled.run(cycles=5, inputs={"clk": 0, "rst": 1})
    count_values = [v for _, v in traces["count"]]
    assert all(v == 0 for v in count_values), f"Expected all 0s, got {count_values}"


def _make_dff() -> Module:
    """D flip-flop: q <= d on posedge clk."""
    return Module(
        name="dff",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("d", Shape(8)), PortDirection.INPUT),
            Port(Signal("q", Shape(8)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(SeqBlock("sync", stmts=(Assign("q", SignalRef(Shape(8), "d")),)),),
    )


class TestDFlipFlop:
    def test_captures_d_on_posedge(self):
        """q should take the value of d on each posedge."""
        m = _make_dff()
        cm = compile_module(m)
        traces = cm.run(cycles=3, inputs={"d": 42})
        q_vals = [v for _, v in traces["q"]]
        assert len(q_vals) == 3
        assert all(v == 42 for v in q_vals)

    def test_different_input(self):
        """With d=0xFF, q should capture 0xFF."""
        m = _make_dff()
        cm = compile_module(m)
        traces = cm.run(cycles=2, inputs={"d": 0xFF})
        q_vals = [v for _, v in traces["q"]]
        assert all(v == 255 for v in q_vals)


def _make_shift_register() -> Module:
    """4-bit shift register: shifts left on each posedge, serial input at LSB."""
    return Module(
        name="shift_reg",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("sin", Shape(1)), PortDirection.INPUT),
            Port(Signal("q", Shape(4)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                "sync",
                stmts=(
                    Assign(
                        "q",
                        Concat(
                            Shape(4),
                            parts=(
                                # Upper 3 bits: q[2:0] shifted left
                                Slice(Shape(3), SignalRef(Shape(4), "q"), low=0, high=3),
                                # LSB: serial input
                                SignalRef(Shape(1), "sin"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestShiftRegister:
    def test_shift_ones(self):
        """Shifting in 1s: q should go 0001, 0011, 0111, 1111."""
        m = _make_shift_register()
        cm = compile_module(m)
        traces = cm.run(cycles=4, inputs={"sin": 1})
        q_vals = [v for _, v in traces["q"]]
        assert q_vals == [0b0001, 0b0011, 0b0111, 0b1111]

    def test_shift_zeros(self):
        """Shifting in 0s from all-zero register stays zero."""
        m = _make_shift_register()
        cm = compile_module(m)
        traces = cm.run(cycles=4, inputs={"sin": 0})
        q_vals = [v for _, v in traces["q"]]
        assert q_vals == [0, 0, 0, 0]


def _make_counter_sync_reset() -> Module:
    """4-bit counter with domain-level synchronous reset."""
    return Module(
        name="counter_srst",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("rst", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(4), init=0), PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(
                "sync",
                clk="clk",
                edge=EdgePolarity.POSEDGE,
                rst="rst",
                rst_style=ResetStyle.SYNC,
                rst_active_high=True,
            ),
        ),
        seq_blocks=(
            SeqBlock(
                "sync",
                stmts=(
                    Assign(
                        "count",
                        Binary(
                            Shape(4),
                            BinaryOp.ADD,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 1),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestCounterSyncReset:
    def test_counts_when_not_reset(self):
        m = _make_counter_sync_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=10, inputs={"rst": 0})
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 10
        for i, v in enumerate(vals):
            assert v == (i + 1) & 0xF, f"cycle {i}: expected {(i + 1) & 0xF}, got {v}"

    def test_held_in_reset(self):
        m = _make_counter_sync_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=5, inputs={"rst": 1})
        vals = [v for _, v in traces["count"]]
        assert all(v == 0 for v in vals), f"Expected all zeros, got {vals}"

    def test_sync_reset_on_clock_edge_only(self):
        """Sync reset only takes effect on the active clock edge."""
        m = _make_counter_sync_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=3, inputs={"rst": 1})
        vals = [v for _, v in traces["count"]]
        assert vals == [0, 0, 0]


def _make_counter_async_reset() -> Module:
    """4-bit counter with asynchronous reset."""
    return Module(
        name="counter_arst",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("rst", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(4), init=0), PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(
                "sync",
                clk="clk",
                edge=EdgePolarity.POSEDGE,
                rst="rst",
                rst_style=ResetStyle.ASYNC,
                rst_active_high=True,
            ),
        ),
        seq_blocks=(
            SeqBlock(
                "sync",
                stmts=(
                    Assign(
                        "count",
                        Binary(
                            Shape(4),
                            BinaryOp.ADD,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 1),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestCounterAsyncReset:
    def test_async_reset_holds_at_init(self):
        """With async rst=1, count should never change from init."""
        m = _make_counter_async_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=5, inputs={"rst": 1})
        vals = [v for _, v in traces["count"]]
        assert all(v == 0 for v in vals) or len(vals) == 0

    def test_counts_when_not_reset(self):
        m = _make_counter_async_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=10, inputs={"rst": 0})
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 10
        for i, v in enumerate(vals):
            assert v == (i + 1) & 0xF


def _make_negedge_counter() -> Module:
    """4-bit counter sensitive to negedge clk."""
    return Module(
        name="neg_counter",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(4), init=0), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.NEGEDGE),),
        seq_blocks=(
            SeqBlock(
                "sync",
                stmts=(
                    Assign(
                        "count",
                        Binary(
                            Shape(4),
                            BinaryOp.ADD,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 1),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestNegedgeCounter:
    def test_increments_on_negedge(self):
        """Counter should increment on falling edge, not rising edge."""
        m = _make_negedge_counter()
        cm = compile_module(m)
        traces = cm.run(cycles=5)
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 5
        for i, v in enumerate(vals):
            assert v == (i + 1) & 0xF


def _make_counter_with_decode() -> Module:
    """4-bit counter + comb decode: zero = (count == 0), max = (count == 15)."""
    return Module(
        name="counter_decode",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(4), init=0), PortDirection.OUTPUT),
            Port(Signal("is_zero", Shape(1), init=1), PortDirection.OUTPUT),
            Port(Signal("is_max", Shape(1), init=0), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                "sync",
                stmts=(
                    Assign(
                        "count",
                        Binary(
                            Shape(4),
                            BinaryOp.ADD,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 1),
                        ),
                    ),
                ),
            ),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        "is_zero",
                        Binary(
                            Shape(1),
                            BinaryOp.EQ,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 0),
                        ),
                    ),
                )
            ),
            CombBlock(
                stmts=(
                    Assign(
                        "is_max",
                        Binary(
                            Shape(1),
                            BinaryOp.EQ,
                            SignalRef(Shape(4), "count"),
                            Const(Shape(4), 15),
                        ),
                    ),
                )
            ),
        ),
    )


class TestCounterWithDecode:
    def test_decode_flags(self):
        """Comb decode updates after seq block on each posedge."""
        m = _make_counter_with_decode()
        cm = compile_module(m)
        traces = cm.run(cycles=16)
        count_vals = [v for _, v in traces["count"]]
        zero_vals = [v for _, v in traces["is_zero"]]
        max_vals = [v for _, v in traces["is_max"]]

        for i, v in enumerate(count_vals):
            assert v == (i + 1) & 0xF

        for i, v in enumerate(zero_vals):
            expected = 1 if (i + 1) & 0xF == 0 else 0
            assert v == expected, f"cycle {i}: is_zero expected {expected}, got {v}"

        for i, v in enumerate(max_vals):
            expected = 1 if (i + 1) & 0xF == 15 else 0
            assert v == expected, f"cycle {i}: is_max expected {expected}, got {v}"


def _make_dual_clock() -> Module:
    """Two counters on different clock domains."""
    return Module(
        name="dual_clock",
        ports=(
            Port(Signal("fast_clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("slow_clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("fast_count", Shape(8), init=0), PortDirection.OUTPUT),
            Port(Signal("slow_count", Shape(8), init=0), PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain("fast", clk="fast_clk", edge=EdgePolarity.POSEDGE),
            ClockDomain("slow", clk="slow_clk", edge=EdgePolarity.POSEDGE),
        ),
        seq_blocks=(
            SeqBlock(
                "fast",
                stmts=(
                    Assign(
                        "fast_count",
                        Binary(
                            Shape(8),
                            BinaryOp.ADD,
                            SignalRef(Shape(8), "fast_count"),
                            Const(Shape(8), 1),
                        ),
                    ),
                ),
            ),
            SeqBlock(
                "slow",
                stmts=(
                    Assign(
                        "slow_count",
                        Binary(
                            Shape(8),
                            BinaryOp.ADD,
                            SignalRef(Shape(8), "slow_count"),
                            Const(Shape(8), 1),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestDualClock:
    def test_different_rates(self):
        """Fast clock runs 2x faster than slow clock."""
        m = _make_dual_clock()
        cm = compile_module(m)
        traces = cm.run(
            cycles=10,
            clock_period=timedelta(microseconds=1),
            clocks={
                "fast": timedelta(microseconds=1),
                "slow": timedelta(microseconds=2),
            },
        )
        fast_vals = [v for _, v in traces["fast_count"]]
        slow_vals = [v for _, v in traces["slow_count"]]

        assert fast_vals[-1] == 10, f"Expected fast_count=10, got {fast_vals[-1]}"
        assert slow_vals[-1] == 5, f"Expected slow_count=5, got {slow_vals[-1]}"


class TestLongRunCounter:
    def test_100_cycle_counter(self):
        """Simulate 4-bit counter with reset for 100 cycles."""
        m = _make_counter_sync_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=100, inputs={"rst": 0})
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 100
        for i, v in enumerate(vals):
            expected = (i + 1) & 0xF
            assert v == expected, f"cycle {i}: expected {expected}, got {v}"

    def test_100_cycles_counting(self):
        """Counter should wrap correctly over 100 cycles."""
        m = _make_counter_sync_reset()
        cm = compile_module(m)
        traces = cm.run(cycles=100, inputs={"rst": 0})
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 100
        for i, v in enumerate(vals):
            assert v == (i + 1) % 16


class TestFourStateSequential:
    def test_dff_four_state(self):
        """D flip-flop in four-state mode should capture d correctly."""
        m = _make_dff()
        cm = compile_module(m, four_state=True)
        traces = cm.run(cycles=3, inputs={"d": 42})
        q_vals = [v for _, v in traces["q"]]
        assert len(q_vals) == 3
        assert all(v == 42 for v in q_vals)

    def test_counter_four_state(self):
        """Counter in four-state mode."""
        m = _make_counter_sync_reset()
        cm = compile_module(m, four_state=True)
        traces = cm.run(cycles=10, inputs={"rst": 0})
        vals = [v for _, v in traces["count"]]
        assert len(vals) == 10
        for i, v in enumerate(vals):
            assert v == (i + 1) & 0xF
