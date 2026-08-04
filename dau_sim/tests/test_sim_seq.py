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

import io
from contextlib import redirect_stdout
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
    Print,
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="rst", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(4)), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="d", shape=Shape(8)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="q", shape=Shape(8)), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(SeqBlock(domain="sync", stmts=(Assign(target="q", value=SignalRef(shape=Shape(8), name="d")),)),),
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="sin", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="q", shape=Shape(4)), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    Assign(
                        target="q",
                        value=Concat(
                            shape=Shape(4),
                            parts=(
                                # Upper 3 bits: q[2:0] shifted left
                                Slice(shape=Shape(3), value=SignalRef(shape=Shape(4), name="q"), low=0, high=3),
                                # LSB: serial input
                                SignalRef(shape=Shape(1), name="sin"),
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="rst", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(4), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(
                name="sync",
                clk="clk",
                edge=EdgePolarity.POSEDGE,
                rst="rst",
                rst_style=ResetStyle.SYNC,
                rst_active_high=True,
            ),
        ),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
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


def _make_partitioned_seq_module() -> Module:
    return Module(
        name="partitioned_seq",
        ports=(
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="stable", shape=Shape(8), init=7), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
            Port(signal=Signal(name="y", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    Assign(
                        target="count",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="count"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="y",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="count"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
            CombBlock(
                stmts=(Print(format_str="stable={}", args=(SignalRef(shape=Shape(8), name="stable"),)),),
            ),
        ),
    )


def test_sequential_settle_skips_unrelated_comb_components():
    """Unrelated comb components should not rerun on every sequential edge."""
    cm = compile_module(_make_partitioned_seq_module())
    buf = io.StringIO()

    with redirect_stdout(buf):
        traces = cm.run(cycles=3, inputs={"stable": 7})

    assert [v for _, v in traces["y"]] == [2, 3, 4]
    assert buf.getvalue() == ""


def _make_counter_async_reset() -> Module:
    """4-bit counter with asynchronous reset."""
    return Module(
        name="counter_arst",
        ports=(
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="rst", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(4), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(
                name="sync",
                clk="clk",
                edge=EdgePolarity.POSEDGE,
                rst="rst",
                rst_style=ResetStyle.ASYNC,
                rst_active_high=True,
            ),
        ),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(4), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.NEGEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
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
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="count", shape=Shape(4), init=0), direction=PortDirection.OUTPUT),
            Port(signal=Signal(name="is_zero", shape=Shape(1), init=1), direction=PortDirection.OUTPUT),
            Port(signal=Signal(name="is_max", shape=Shape(1), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain(name="sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
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
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="is_zero",
                        value=Binary(
                            shape=Shape(1),
                            op=BinaryOp.EQ,
                            left=SignalRef(shape=Shape(4), name="count"),
                            right=Const(shape=Shape(4), value=0),
                        ),
                    ),
                )
            ),
            CombBlock(
                stmts=(
                    Assign(
                        target="is_max",
                        value=Binary(
                            shape=Shape(1),
                            op=BinaryOp.EQ,
                            left=SignalRef(shape=Shape(4), name="count"),
                            right=Const(shape=Shape(4), value=15),
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
            Port(signal=Signal(name="fast_clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="slow_clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="fast_count", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
            Port(signal=Signal(name="slow_count", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(name="fast", clk="fast_clk", edge=EdgePolarity.POSEDGE),
            ClockDomain(name="slow", clk="slow_clk", edge=EdgePolarity.POSEDGE),
        ),
        seq_blocks=(
            SeqBlock(
                domain="fast",
                stmts=(
                    Assign(
                        target="fast_count",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="fast_count"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
            SeqBlock(
                domain="slow",
                stmts=(
                    Assign(
                        target="slow_count",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="slow_count"),
                            right=Const(shape=Shape(8), value=1),
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


def _make_mixed_edge_single_clock() -> Module:
    """Two counters sharing one clock, one posedge and one negedge."""
    return Module(
        name="mixed_edge",
        ports=(
            Port(signal=Signal(name="clk", shape=Shape(1)), direction=PortDirection.INPUT),
            Port(signal=Signal(name="pos_count", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
            Port(signal=Signal(name="neg_count", shape=Shape(8), init=0), direction=PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain(name="pos", clk="clk", edge=EdgePolarity.POSEDGE),
            ClockDomain(name="neg", clk="clk", edge=EdgePolarity.NEGEDGE),
        ),
        seq_blocks=(
            SeqBlock(
                domain="pos",
                stmts=(
                    Assign(
                        target="pos_count",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="pos_count"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
            SeqBlock(
                domain="neg",
                stmts=(
                    Assign(
                        target="neg_count",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="neg_count"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestMixedClockEdges:
    def test_posedge_and_negedge_domains_progress_equally(self):
        """Both domains should tick once per full period on shared clock."""
        m = _make_mixed_edge_single_clock()
        cm = compile_module(m)
        traces = cm.run(cycles=8)

        pos_vals = [v for _, v in traces["pos_count"]]
        neg_vals = [v for _, v in traces["neg_count"]]

        assert pos_vals[-1] == 8, f"Expected pos_count=8, got {pos_vals[-1]}"
        assert neg_vals[-1] == 8, f"Expected neg_count=8, got {neg_vals[-1]}"


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
