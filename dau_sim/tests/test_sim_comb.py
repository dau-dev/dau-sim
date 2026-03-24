"""Tests for combinational simulation: IR → compile → run → verify traces.

Circuits tested:
- Simple adder and mux (basic end-to-end)
- Ripple-carry adder (4-bit)
- Multiplexer tree (4:1)
- ALU (8-bit, 4 ops)
- Priority encoder (8→3)
- Four-state simulation mode
"""

from dau_sim.compiler import compile_module
from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    IfElse,
    Module,
    Mux,
    Port,
    PortDirection,
    Shape,
    Signal,
    SignalRef,
    Slice,
    Switch,
)

# ═══════════════════════════════════════════════════════════════════
# Basic combinational end-to-end
# ═══════════════════════════════════════════════════════════════════


def test_compile_combinational_adder():
    """Pure combinational: a + b = sum, driven each tick."""
    m = Module(
        name="adder",
        ports=(
            Port(Signal("a", Shape(8)), PortDirection.INPUT),
            Port(Signal("b", Shape(8)), PortDirection.INPUT),
            Port(Signal("sum", Shape(8)), PortDirection.OUTPUT),
        ),
        clock_domains=(
            ClockDomain("sync", clk="a"),  # need a clock domain for ticking
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="sum",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="a"),
                            right=SignalRef(shape=Shape(8), name="b"),
                        ),
                    ),
                ),
            ),
        ),
    )

    compiled = compile_module(m)
    traces = compiled.run(cycles=3, inputs={"a": 10, "b": 20})

    sum_values = [v for _, v in traces["sum"]]
    assert all(v == 30 for v in sum_values), f"Expected all 30, got {sum_values}"


def test_compile_mux_module():
    """Mux: out = sel ? a : b."""
    m = Module(
        name="mux2",
        ports=(
            Port(Signal("sel", Shape(1)), PortDirection.INPUT),
            Port(Signal("a", Shape(8)), PortDirection.INPUT),
            Port(Signal("b", Shape(8)), PortDirection.INPUT),
            Port(Signal("out", Shape(8)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="sel"),),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="out",
                        value=Mux(
                            shape=Shape(8),
                            sel=SignalRef(shape=Shape(1), name="sel"),
                            if_true=SignalRef(shape=Shape(8), name="a"),
                            if_false=SignalRef(shape=Shape(8), name="b"),
                        ),
                    ),
                ),
            ),
        ),
    )

    # sel=1 → out=a=42
    compiled = compile_module(m)
    traces = compiled.run(cycles=1, inputs={"sel": 1, "a": 42, "b": 99})
    out_vals = [v for _, v in traces["out"]]
    assert out_vals[0] == 42

    # sel=0 → out=b=99
    traces = compiled.run(cycles=1, inputs={"sel": 0, "a": 42, "b": 99})
    out_vals = [v for _, v in traces["out"]]
    assert out_vals[0] == 99


# ═══════════════════════════════════════════════════════════════════
# Ripple-carry adder (4-bit)
# ═══════════════════════════════════════════════════════════════════


def _make_ripple_carry_adder() -> Module:
    """4-bit ripple-carry adder: sum = a + b, cout = carry out.

    Expressed as combinational logic using bit slices and full-adder chain.
    For simplicity, we use the ADD operator directly with proper widths.
    """
    return Module(
        name="adder4",
        ports=(
            Port(Signal("a", Shape(4)), PortDirection.INPUT),
            Port(Signal("b", Shape(4)), PortDirection.INPUT),
            Port(Signal("sum", Shape(4)), PortDirection.OUTPUT),
            Port(Signal("cout", Shape(1)), PortDirection.OUTPUT),
        ),
        signals=(
            Signal("wide_sum", Shape(5)),  # 5-bit intermediate to capture carry
        ),
        clock_domains=(ClockDomain("sync", clk="a"),),
        comb_blocks=(
            # Block 0: wide_sum = a + b (5-bit)
            CombBlock(
                stmts=(
                    Assign(
                        target="wide_sum",
                        value=Binary(
                            shape=Shape(5),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(4), name="a"),
                            right=SignalRef(shape=Shape(4), name="b"),
                        ),
                    ),
                )
            ),
            # Block 1: sum = wide_sum[0:4]
            CombBlock(
                stmts=(
                    Assign(
                        target="sum",
                        value=Slice(
                            shape=Shape(4),
                            value=SignalRef(shape=Shape(5), name="wide_sum"),
                            low=0,
                            high=4,
                        ),
                    ),
                )
            ),
            # Block 2: cout = wide_sum[4]
            CombBlock(
                stmts=(
                    Assign(
                        target="cout",
                        value=Slice(
                            shape=Shape(1),
                            value=SignalRef(shape=Shape(5), name="wide_sum"),
                            low=4,
                            high=5,
                        ),
                    ),
                )
            ),
        ),
    )


class TestRippleCarryAdder:
    def test_no_carry(self):
        """3 + 4 = 7, no carry."""
        m = _make_ripple_carry_adder()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 3, "b": 4})
        assert traces["sum"][0][1] == 7
        assert traces["cout"][0][1] == 0

    def test_with_carry(self):
        """15 + 1 = 0 with carry."""
        m = _make_ripple_carry_adder()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 15, "b": 1})
        assert traces["sum"][0][1] == 0
        assert traces["cout"][0][1] == 1

    def test_max_plus_max(self):
        """15 + 15 = 14 with carry (30 = 0b11110)."""
        m = _make_ripple_carry_adder()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 15, "b": 15})
        assert traces["sum"][0][1] == 14  # 30 & 0xF
        assert traces["cout"][0][1] == 1

    def test_dependency_order(self):
        """Verify that comb blocks are evaluated in correct dependency order
        (wide_sum before sum/cout)."""
        m = _make_ripple_carry_adder()
        c = compile_module(m)
        # The first assignment in order should write wide_sum
        assert "wide_sum" in c._comb_order[0].writes


# ═══════════════════════════════════════════════════════════════════
# Multiplexer tree (4:1)
# ═══════════════════════════════════════════════════════════════════


def _make_mux_tree() -> Module:
    """4:1 multiplexer tree using two 2:1 muxes and a final 2:1 mux.

    Inputs: in0..in3 (8-bit), sel (2-bit)
    Output: out (8-bit)
    """
    return Module(
        name="mux4",
        ports=(
            Port(Signal("in0", Shape(8)), PortDirection.INPUT),
            Port(Signal("in1", Shape(8)), PortDirection.INPUT),
            Port(Signal("in2", Shape(8)), PortDirection.INPUT),
            Port(Signal("in3", Shape(8)), PortDirection.INPUT),
            Port(Signal("sel", Shape(2)), PortDirection.INPUT),
            Port(Signal("out", Shape(8)), PortDirection.OUTPUT),
        ),
        signals=(
            Signal("mux_lo", Shape(8)),  # mux of in0/in1
            Signal("mux_hi", Shape(8)),  # mux of in2/in3
        ),
        clock_domains=(ClockDomain("sync", clk="in0"),),
        comb_blocks=(
            # Stage 1a: mux_lo = sel[0] ? in1 : in0
            CombBlock(
                stmts=(
                    Assign(
                        target="mux_lo",
                        value=Mux(
                            shape=Shape(8),
                            sel=Slice(shape=Shape(1), value=SignalRef(shape=Shape(2), name="sel"), low=0, high=1),
                            if_true=SignalRef(shape=Shape(8), name="in1"),
                            if_false=SignalRef(shape=Shape(8), name="in0"),
                        ),
                    ),
                )
            ),
            # Stage 1b: mux_hi = sel[0] ? in3 : in2
            CombBlock(
                stmts=(
                    Assign(
                        target="mux_hi",
                        value=Mux(
                            shape=Shape(8),
                            sel=Slice(shape=Shape(1), value=SignalRef(shape=Shape(2), name="sel"), low=0, high=1),
                            if_true=SignalRef(shape=Shape(8), name="in3"),
                            if_false=SignalRef(shape=Shape(8), name="in2"),
                        ),
                    ),
                )
            ),
            # Stage 2: out = sel[1] ? mux_hi : mux_lo
            CombBlock(
                stmts=(
                    Assign(
                        target="out",
                        value=Mux(
                            shape=Shape(8),
                            sel=Slice(shape=Shape(1), value=SignalRef(shape=Shape(2), name="sel"), low=1, high=2),
                            if_true=SignalRef(shape=Shape(8), name="mux_hi"),
                            if_false=SignalRef(shape=Shape(8), name="mux_lo"),
                        ),
                    ),
                )
            ),
        ),
    )


class TestMuxTree:
    def test_sel_0(self):
        m = _make_mux_tree()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"in0": 10, "in1": 20, "in2": 30, "in3": 40, "sel": 0})
        assert traces["out"][0][1] == 10

    def test_sel_1(self):
        m = _make_mux_tree()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"in0": 10, "in1": 20, "in2": 30, "in3": 40, "sel": 1})
        assert traces["out"][0][1] == 20

    def test_sel_2(self):
        m = _make_mux_tree()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"in0": 10, "in1": 20, "in2": 30, "in3": 40, "sel": 2})
        assert traces["out"][0][1] == 30

    def test_sel_3(self):
        m = _make_mux_tree()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"in0": 10, "in1": 20, "in2": 30, "in3": 40, "sel": 3})
        assert traces["out"][0][1] == 40


# ═══════════════════════════════════════════════════════════════════
# ALU (8-bit, 4 operations)
# ═══════════════════════════════════════════════════════════════════


def _make_alu() -> Module:
    """8-bit ALU with 4 operations selected by 2-bit op code.

    op=0: ADD, op=1: SUB, op=2: AND, op=3: OR
    """
    return Module(
        name="alu",
        ports=(
            Port(Signal("a", Shape(8)), PortDirection.INPUT),
            Port(Signal("b", Shape(8)), PortDirection.INPUT),
            Port(Signal("op", Shape(2)), PortDirection.INPUT),
            Port(Signal("result", Shape(8)), PortDirection.OUTPUT),
            Port(Signal("zero", Shape(1)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="a"),),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Switch(
                        test=SignalRef(shape=Shape(2), name="op"),
                        cases=(
                            (
                                0,
                                (
                                    Assign(
                                        target="result",
                                        value=Binary(
                                            shape=Shape(8),
                                            op=BinaryOp.ADD,
                                            left=SignalRef(shape=Shape(8), name="a"),
                                            right=SignalRef(shape=Shape(8), name="b"),
                                        ),
                                    ),
                                ),
                            ),
                            (
                                1,
                                (
                                    Assign(
                                        target="result",
                                        value=Binary(
                                            shape=Shape(8),
                                            op=BinaryOp.SUB,
                                            left=SignalRef(shape=Shape(8), name="a"),
                                            right=SignalRef(shape=Shape(8), name="b"),
                                        ),
                                    ),
                                ),
                            ),
                            (
                                2,
                                (
                                    Assign(
                                        target="result",
                                        value=Binary(
                                            shape=Shape(8),
                                            op=BinaryOp.AND,
                                            left=SignalRef(shape=Shape(8), name="a"),
                                            right=SignalRef(shape=Shape(8), name="b"),
                                        ),
                                    ),
                                ),
                            ),
                            (
                                3,
                                (
                                    Assign(
                                        target="result",
                                        value=Binary(
                                            shape=Shape(8),
                                            op=BinaryOp.OR,
                                            left=SignalRef(shape=Shape(8), name="a"),
                                            right=SignalRef(shape=Shape(8), name="b"),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            ),
            # Zero flag: zero = (result == 0)
            CombBlock(
                stmts=(
                    Assign(
                        target="zero",
                        value=Binary(
                            shape=Shape(1),
                            op=BinaryOp.EQ,
                            left=SignalRef(shape=Shape(8), name="result"),
                            right=Const(shape=Shape(8), value=0),
                        ),
                    ),
                )
            ),
        ),
    )


class TestALU:
    def test_add(self):
        m = _make_alu()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 100, "b": 50, "op": 0})
        assert traces["result"][0][1] == 150

    def test_sub(self):
        m = _make_alu()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 100, "b": 50, "op": 1})
        assert traces["result"][0][1] == 50

    def test_and(self):
        m = _make_alu()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 0xAA, "b": 0x55, "op": 2})
        assert traces["result"][0][1] == 0x00

    def test_or(self):
        m = _make_alu()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 0xAA, "b": 0x55, "op": 3})
        assert traces["result"][0][1] == 0xFF

    def test_zero_flag(self):
        m = _make_alu()
        c = compile_module(m)
        # 100 - 100 = 0 → zero=1
        traces = c.run(cycles=1, inputs={"a": 100, "b": 100, "op": 1})
        assert traces["result"][0][1] == 0
        assert traces["zero"][0][1] == 1

    def test_nonzero_flag(self):
        m = _make_alu()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"a": 100, "b": 50, "op": 0})
        assert traces["zero"][0][1] == 0


# ═══════════════════════════════════════════════════════════════════
# Priority encoder (8→3)
# ═══════════════════════════════════════════════════════════════════


def _make_priority_encoder() -> Module:
    """8-to-3 priority encoder.

    Outputs the index of the highest set bit in the 8-bit input.
    valid=1 if any bit is set, valid=0 otherwise.
    """
    return Module(
        name="prienc",
        ports=(
            Port(Signal("inp", Shape(8)), PortDirection.INPUT),
            Port(Signal("idx", Shape(3)), PortDirection.OUTPUT),
            Port(Signal("valid", Shape(1)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="inp"),),
        comb_blocks=(
            CombBlock(
                stmts=(
                    # Default: idx=0, valid=0
                    Assign(target="idx", value=Const(shape=Shape(3), value=0)),
                    Assign(target="valid", value=Const(shape=Shape(1), value=0)),
                    # Priority chain: highest bit wins
                    IfElse(
                        cond=Binary(
                            shape=Shape(1),
                            op=BinaryOp.NE,
                            left=SignalRef(shape=Shape(8), name="inp"),
                            right=Const(shape=Shape(8), value=0),
                        ),
                        then_body=(
                            Assign(target="valid", value=Const(shape=Shape(1), value=1)),
                            IfElse(
                                cond=Slice(shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=7, high=8),
                                then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=7)),),
                                else_body=(
                                    IfElse(
                                        cond=Slice(shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=6, high=7),
                                        then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=6)),),
                                        else_body=(
                                            IfElse(
                                                cond=Slice(shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=5, high=6),
                                                then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=5)),),
                                                else_body=(
                                                    IfElse(
                                                        cond=Slice(shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=4, high=5),
                                                        then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=4)),),
                                                        else_body=(
                                                            IfElse(
                                                                cond=Slice(
                                                                    shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=3, high=4
                                                                ),
                                                                then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=3)),),
                                                                else_body=(
                                                                    IfElse(
                                                                        cond=Slice(
                                                                            shape=Shape(1), value=SignalRef(shape=Shape(8), name="inp"), low=2, high=3
                                                                        ),
                                                                        then_body=(Assign(target="idx", value=Const(shape=Shape(3), value=2)),),
                                                                        else_body=(
                                                                            IfElse(
                                                                                cond=Slice(
                                                                                    shape=Shape(1),
                                                                                    value=SignalRef(shape=Shape(8), name="inp"),
                                                                                    low=1,
                                                                                    high=2,
                                                                                ),
                                                                                then_body=(
                                                                                    Assign(target="idx", value=Const(shape=Shape(3), value=1)),
                                                                                ),
                                                                                else_body=(
                                                                                    Assign(target="idx", value=Const(shape=Shape(3), value=0)),
                                                                                ),
                                                                            ),
                                                                        ),
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
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
            ),
        ),
    )


class TestPriorityEncoder:
    def test_no_bits_set(self):
        m = _make_priority_encoder()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"inp": 0})
        assert traces["valid"][0][1] == 0

    def test_single_bit(self):
        m = _make_priority_encoder()
        c = compile_module(m)
        for bit in range(8):
            traces = c.run(cycles=1, inputs={"inp": 1 << bit})
            assert traces["valid"][0][1] == 1, f"bit={bit}"
            assert traces["idx"][0][1] == bit, f"bit={bit}: got {traces['idx'][0][1]}"

    def test_highest_wins(self):
        """With multiple bits set, highest index wins."""
        m = _make_priority_encoder()
        c = compile_module(m)
        traces = c.run(cycles=1, inputs={"inp": 0b10000001})  # bits 7 and 0
        assert traces["idx"][0][1] == 7
        assert traces["valid"][0][1] == 1


# ═══════════════════════════════════════════════════════════════════
# Four-State Simulation
# ═══════════════════════════════════════════════════════════════════


class TestFourStateSimulation:
    def test_adder_four_state(self):
        """Same adder, but compiled in four-state mode."""
        m = _make_ripple_carry_adder()
        c = compile_module(m, four_state=True)
        traces = c.run(cycles=1, inputs={"a": 3, "b": 4})
        assert traces["sum"][0][1] == 7
        assert traces["cout"][0][1] == 0

    def test_alu_four_state(self):
        m = _make_alu()
        c = compile_module(m, four_state=True)
        traces = c.run(cycles=1, inputs={"a": 100, "b": 50, "op": 0})
        assert traces["result"][0][1] == 150
