from dau_sim.compiler.eval import eval_expr, mask_value
from dau_sim.compiler.eval4 import eval_expr_4
from dau_sim.ir import (
    Binary,
    BinaryOp,
    Concat,
    Const,
    FourState,
    Mux,
    Shape,
    SignalRef,
    Slice,
    Unary,
    UnaryOp,
)


def test_eval_const():
    c = Const(shape=Shape(8), value=42)
    assert eval_expr(c, {}) == 42


def test_eval_const_truncation():
    # 256 should wrap to 0 in 8-bit unsigned
    c = Const(shape=Shape(8), value=256)
    assert eval_expr(c, {}) == 0

    # -1 in 8-bit unsigned = 255
    c = Const(shape=Shape(8), value=-1)
    assert eval_expr(c, {}) == 255

    # -1 in 8-bit signed = -1
    c = Const(shape=Shape(8, signed=True), value=-1)
    assert eval_expr(c, {}) == -1


def test_eval_signal_ref():
    r = SignalRef(shape=Shape(8), name="x")
    assert eval_expr(r, {"x": 42}) == 42


def test_eval_add():
    a = Const(shape=Shape(8), value=100)
    b = Const(shape=Shape(8), value=200)
    add = Binary(shape=Shape(8), op=BinaryOp.ADD, left=a, right=b)
    # 100 + 200 = 300, truncated to 8 bits = 44
    assert eval_expr(add, {}) == 44


def test_eval_sub():
    a = Const(shape=Shape(8), value=10)
    b = Const(shape=Shape(8), value=20)
    sub = Binary(shape=Shape(8), op=BinaryOp.SUB, left=a, right=b)
    # 10 - 20 = -10, in 8-bit unsigned = 246
    assert eval_expr(sub, {}) == 246


def test_eval_bitwise():
    a = Const(shape=Shape(8), value=0xAA)
    b = Const(shape=Shape(8), value=0x55)
    and_expr = Binary(shape=Shape(8), op=BinaryOp.AND, left=a, right=b)
    assert eval_expr(and_expr, {}) == 0x00

    or_expr = Binary(shape=Shape(8), op=BinaryOp.OR, left=a, right=b)
    assert eval_expr(or_expr, {}) == 0xFF

    xor_expr = Binary(shape=Shape(8), op=BinaryOp.XOR, left=a, right=b)
    assert eval_expr(xor_expr, {}) == 0xFF


def test_eval_comparison():
    a = Const(shape=Shape(8), value=5)
    b = Const(shape=Shape(8), value=10)

    eq = Binary(shape=Shape(1), op=BinaryOp.EQ, left=a, right=b)
    assert eval_expr(eq, {}) == 0

    lt = Binary(shape=Shape(1), op=BinaryOp.LT, left=a, right=b)
    assert eval_expr(lt, {}) == 1

    eq_self = Binary(shape=Shape(1), op=BinaryOp.EQ, left=a, right=a)
    assert eval_expr(eq_self, {}) == 1


def test_eval_mux():
    sel_1 = Const(shape=Shape(1), value=1)
    sel_0 = Const(shape=Shape(1), value=0)
    a = Const(shape=Shape(8), value=10)
    b = Const(shape=Shape(8), value=20)

    m1 = Mux(shape=Shape(8), sel=sel_1, if_true=a, if_false=b)
    assert eval_expr(m1, {}) == 10

    m0 = Mux(shape=Shape(8), sel=sel_0, if_true=a, if_false=b)
    assert eval_expr(m0, {}) == 20


def test_eval_concat():
    hi = Const(shape=Shape(4), value=0xA)
    lo = Const(shape=Shape(4), value=0x5)
    c = Concat(shape=Shape(8), parts=(hi, lo))
    assert eval_expr(c, {}) == 0xA5


def test_eval_slice():
    val = Const(shape=Shape(8), value=0xA5)
    lo = Slice(shape=Shape(4), value=val, low=0, high=4)
    assert eval_expr(lo, {}) == 0x5

    hi = Slice(shape=Shape(4), value=val, low=4, high=8)
    assert eval_expr(hi, {}) == 0xA


def test_eval_unary_not():
    a = Const(shape=Shape(8), value=0xAA)
    n = Unary(shape=Shape(8), op=UnaryOp.NOT, operand=a)
    assert eval_expr(n, {}) == 0x55


def test_eval_reduction():
    # RED_AND of 0xFF (8 bits, all 1s) = 1
    a = Const(shape=Shape(8), value=0xFF)
    ra = Unary(shape=Shape(1), op=UnaryOp.RED_AND, operand=a)
    assert eval_expr(ra, {}) == 1

    # RED_AND of 0xFE (8 bits, one 0) = 0
    b = Const(shape=Shape(8), value=0xFE)
    rb = Unary(shape=Shape(1), op=UnaryOp.RED_AND, operand=b)
    assert eval_expr(rb, {}) == 0


def test_mask_value():
    assert mask_value(256, Shape(8)) == 0
    assert mask_value(-1, Shape(8)) == 255
    assert mask_value(-1, Shape(8, signed=True)) == -1
    assert mask_value(127, Shape(8, signed=True)) == 127
    assert mask_value(128, Shape(8, signed=True)) == -128


class TestFourState:
    def test_from_int(self):
        v = FourState.from_int(42, Shape(8))
        assert v.aval == 42
        assert v.bval == 0
        assert v.is_fully_defined
        assert v.to_int == 42

    def test_x_value(self):
        v = FourState.x(Shape(8))
        assert v.bval == 0xFF
        assert v.has_unknown
        assert v.to_int is None

    def test_z_value(self):
        v = FourState.z(Shape(4))
        assert v.aval == 0
        assert v.bval == 0xF
        assert v.has_unknown

    def test_width_normalization(self):
        # Value wider than shape gets truncated
        v = FourState(shape=Shape(4), aval=0xFF, bval=0)
        assert v.aval == 0xF
        assert v.bval == 0

    def test_repr_defined(self):
        v = FourState.from_int(5, Shape(4))
        assert "5" in repr(v)

    def test_repr_x(self):
        v = FourState(shape=Shape(4), aval=0b1010, bval=0b0110)
        s = repr(v)
        # bits 3..0: a=1010, b=0110
        # bit 3: a=1, b=0 → 1
        # bit 2: a=0, b=1 → z
        # bit 1: a=1, b=1 → x
        # bit 0: a=0, b=0 → 0
        assert s == "1zx0"

    def test_signed_to_int(self):
        v = FourState.from_int(-1, Shape(8, signed=True))
        assert v.aval == 0xFF
        assert v.to_int == -1


class TestEval4:
    def test_const(self):
        c = Const(shape=Shape(8), value=42)
        r = eval_expr_4(c, {})
        assert r.to_int == 42

    def test_signal_ref(self):
        signals = {"x": FourState.from_int(10, Shape(8))}
        r = eval_expr_4(SignalRef(shape=Shape(8), name="x"), signals)
        assert r.to_int == 10

    def test_add_defined(self):
        a = Const(shape=Shape(8), value=100)
        b = Const(shape=Shape(8), value=200)
        add = Binary(shape=Shape(8), op=BinaryOp.ADD, left=a, right=b)
        r = eval_expr_4(add, {})
        assert r.to_int == 44  # 300 & 0xFF

    def test_add_with_x(self):
        """Any X in arithmetic → all-X result."""
        signals = {
            "a": FourState.x(Shape(8)),
            "b": FourState.from_int(5, Shape(8)),
        }
        add = Binary(
            shape=Shape(8),
            op=BinaryOp.ADD,
            left=SignalRef(shape=Shape(8), name="a"),
            right=SignalRef(shape=Shape(8), name="b"),
        )
        r = eval_expr_4(add, signals)
        assert r.has_unknown

    def test_bitwise_and_x_propagation(self):
        """0 & X = 0 (known); 1 & X = X."""
        signals = {
            "a": FourState.from_int(0x0F, Shape(8)),  # 0000_1111
            "b": FourState.x(Shape(8)),  # xxxx_xxxx
        }
        and_expr = Binary(
            shape=Shape(8),
            op=BinaryOp.AND,
            left=SignalRef(shape=Shape(8), name="a"),
            right=SignalRef(shape=Shape(8), name="b"),
        )
        r = eval_expr_4(and_expr, signals)
        # Upper 4 bits: 0 & X = 0 (defined)
        # Lower 4 bits: 1 & X = X
        assert (r.aval & 0xF0) == 0
        assert (r.bval & 0xF0) == 0
        assert (r.bval & 0x0F) == 0x0F

    def test_bitwise_or_x_propagation(self):
        """1 | X = 1 (known); 0 | X = X."""
        signals = {
            "a": FourState.from_int(0xF0, Shape(8)),  # 1111_0000
            "b": FourState.x(Shape(8)),  # xxxx_xxxx
        }
        or_expr = Binary(
            shape=Shape(8),
            op=BinaryOp.OR,
            left=SignalRef(shape=Shape(8), name="a"),
            right=SignalRef(shape=Shape(8), name="b"),
        )
        r = eval_expr_4(or_expr, signals)
        # Upper 4 bits: 1 | X = 1 (defined)
        # Lower 4 bits: 0 | X = X
        assert (r.aval & 0xF0) == 0xF0
        assert (r.bval & 0xF0) == 0
        assert (r.bval & 0x0F) == 0x0F

    def test_not_x(self):
        """~X = X; ~defined = defined."""
        signals = {
            "a": FourState(shape=Shape(4), aval=0b1010, bval=0b0110),
            # bit3=1, bit2=z, bit1=x, bit0=0
        }
        not_expr = Unary(
            shape=Shape(4),
            op=UnaryOp.NOT,
            operand=SignalRef(shape=Shape(4), name="a"),
        )
        r = eval_expr_4(not_expr, signals)
        # bit3: ~1 = 0 (defined)
        # bit2: ~z = x
        # bit1: ~x = x
        # bit0: ~0 = 1 (defined)
        assert (r.bval >> 2) & 1 == 1  # bit 2 is X
        assert (r.bval >> 1) & 1 == 1  # bit 1 is X
        assert (r.bval >> 3) & 1 == 0  # bit 3 defined
        assert (r.bval >> 0) & 1 == 0  # bit 0 defined

    def test_comparison_with_x(self):
        """Comparing anything with X → result is X."""
        signals = {"a": FourState.x(Shape(8))}
        eq = Binary(
            shape=Shape(1),
            op=BinaryOp.EQ,
            left=SignalRef(shape=Shape(8), name="a"),
            right=Const(shape=Shape(8), value=5),
        )
        r = eval_expr_4(eq, signals)
        assert r.has_unknown

    def test_mux_sel_x(self):
        """Mux with X selector: bits that agree keep value, others → X."""
        signals = {
            "sel": FourState.x(Shape(1)),
            "a": FourState.from_int(0xFF, Shape(8)),
            "b": FourState.from_int(0xFF, Shape(8)),
        }
        m = Mux(
            shape=Shape(8),
            sel=SignalRef(shape=Shape(1), name="sel"),
            if_true=SignalRef(shape=Shape(8), name="a"),
            if_false=SignalRef(shape=Shape(8), name="b"),
        )
        r = eval_expr_4(m, signals)
        # Both arms are 0xFF → result should be 0xFF (defined)
        assert r.to_int == 0xFF

    def test_mux_sel_x_divergent(self):
        """Mux with X selector and different arms → X where they differ."""
        signals = {
            "sel": FourState.x(Shape(1)),
            "a": FourState.from_int(0xF0, Shape(8)),
            "b": FourState.from_int(0x0F, Shape(8)),
        }
        m = Mux(
            shape=Shape(8),
            sel=SignalRef(shape=Shape(1), name="sel"),
            if_true=SignalRef(shape=Shape(8), name="a"),
            if_false=SignalRef(shape=Shape(8), name="b"),
        )
        r = eval_expr_4(m, signals)
        # All 8 bits differ → all X
        assert r.bval == 0xFF

    def test_concat_4(self):
        hi = Const(shape=Shape(4), value=0xA)
        lo = Const(shape=Shape(4), value=0x5)
        c = Concat(shape=Shape(8), parts=(hi, lo))
        r = eval_expr_4(c, {})
        assert r.to_int == 0xA5

    def test_slice_4(self):
        c = Const(shape=Shape(8), value=0xA5)
        lo = Slice(shape=Shape(4), value=c, low=0, high=4)
        r = eval_expr_4(lo, {})
        assert r.to_int == 0x5

    def test_logic_or_with_x(self):
        """1 || X = 1 (known nonzero)."""
        signals = {
            "a": FourState.from_int(1, Shape(1)),
            "b": FourState.x(Shape(1)),
        }
        lor = Binary(
            shape=Shape(1),
            op=BinaryOp.LOGIC_OR,
            left=SignalRef(shape=Shape(1), name="a"),
            right=SignalRef(shape=Shape(1), name="b"),
        )
        r = eval_expr_4(lor, signals)
        assert r.to_int == 1

    def test_logic_and_with_zero(self):
        """0 && X = 0 (known zero)."""
        signals = {
            "a": FourState.from_int(0, Shape(1)),
            "b": FourState.x(Shape(1)),
        }
        land = Binary(
            shape=Shape(1),
            op=BinaryOp.LOGIC_AND,
            left=SignalRef(shape=Shape(1), name="a"),
            right=SignalRef(shape=Shape(1), name="b"),
        )
        r = eval_expr_4(land, signals)
        assert r.to_int == 0
