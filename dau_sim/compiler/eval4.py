"""Four-state expression evaluator.

Evaluates IR expression trees with correct IEEE 1364/1800 four-state semantics.
Values are ``FourState`` objects carrying (aval, bval) bitmaps.

X/Z propagation rules (per IEEE 1800-2017):
  - Bitwise ops: X/Z in any input bit → X in output bit
  - Arithmetic: any X/Z bit in either operand → all-X result
  - Comparisons: any X/Z → result is X (represented as X in 1-bit)
  - Logical ops: partial X rules (e.g., 1 || X = 1)
"""

from __future__ import annotations

from dau_sim.ir.expr import (
    Binary,
    BinaryOp,
    Concat,
    Const,
    Expr,
    Mux,
    SignalRef,
    Slice,
    Unary,
    UnaryOp,
)
from dau_sim.ir.types import FourState, Shape


def eval_expr_4(expr: Expr, signals: dict[str, FourState]) -> FourState:
    """Evaluate an expression tree under four-state semantics."""
    if isinstance(expr, Const):
        return FourState.from_int(expr.value, expr.shape)

    if isinstance(expr, SignalRef):
        return signals[expr.name]

    if isinstance(expr, Unary):
        a = eval_expr_4(expr.operand, signals)
        return _eval_unary_4(expr.op, a, expr.shape)

    if isinstance(expr, Binary):
        left = eval_expr_4(expr.left, signals)
        right = eval_expr_4(expr.right, signals)
        return _eval_binary_4(expr.op, left, right, expr.shape)

    if isinstance(expr, Mux):
        sel = eval_expr_4(expr.sel, signals)
        if sel.has_unknown:
            # sel is X/Z: merge both arms into X where they differ
            t = eval_expr_4(expr.if_true, signals)
            f = eval_expr_4(expr.if_false, signals)
            return _mux_x(t, f, expr.shape)
        if sel.aval:
            return _refit(eval_expr_4(expr.if_true, signals), expr.shape)
        return _refit(eval_expr_4(expr.if_false, signals), expr.shape)

    if isinstance(expr, Concat):
        result_a = 0
        result_b = 0
        for part in expr.parts:
            v = eval_expr_4(part, signals)
            w = part.shape.width
            mask = (1 << w) - 1
            result_a = (result_a << w) | (v.aval & mask)
            result_b = (result_b << w) | (v.bval & mask)
        return FourState(shape=expr.shape, aval=result_a, bval=result_b)

    if isinstance(expr, Slice):
        v = eval_expr_4(expr.value, signals)
        w = expr.high - expr.low
        slice_mask = (1 << w) - 1
        aval = (v.aval >> expr.low) & slice_mask
        bval = (v.bval >> expr.low) & slice_mask
        return FourState(shape=expr.shape, aval=aval, bval=bval)

    raise TypeError(f"Unknown expression type: {type(expr).__name__}")


def _refit(v: FourState, shape: Shape) -> FourState:
    """Truncate or zero-extend a FourState to a new shape."""
    return FourState(shape=shape, aval=v.aval, bval=v.bval)


def _all_x(shape: Shape) -> FourState:
    return FourState.x(shape)


def _mux_x(t: FourState, f: FourState, shape: Shape) -> FourState:
    """When mux selector is X: bits that agree keep their value, others → X."""
    agree = ~(t.aval ^ f.aval) & ~(t.bval | f.bval)
    mask = (1 << shape.width) - 1
    agree &= mask
    bval = ~agree & mask
    aval = t.aval & agree
    return FourState(shape=shape, aval=aval, bval=bval)


def _eval_unary_4(op: UnaryOp, a: FourState, out: Shape) -> FourState:
    if a.has_unknown:
        return _unary_with_x(op, a, out)

    v = a.aval
    if op == UnaryOp.NOT:
        return FourState.from_int(~v, out)
    if op == UnaryOp.NEG:
        return FourState.from_int(-_to_signed(v, a.shape), out)
    if op == UnaryOp.BOOL:
        return FourState.from_int(1 if v else 0, out)
    if op == UnaryOp.RED_AND:
        all_ones = (1 << a.shape.width) - 1
        return FourState.from_int(1 if (v & all_ones) == all_ones else 0, out)
    if op == UnaryOp.RED_OR:
        return FourState.from_int(1 if v else 0, out)
    if op == UnaryOp.RED_XOR:
        val = v & ((1 << a.shape.width) - 1)
        return FourState.from_int(bin(val).count("1") & 1, out)
    raise ValueError(f"Unknown unary op: {op}")


def _unary_with_x(op: UnaryOp, a: FourState, out: Shape) -> FourState:
    """Unary op when operand has X/Z bits."""
    if op == UnaryOp.NOT:
        # Bitwise NOT: X/Z bits → X
        aval = (~a.aval) & ~a.bval
        bval = a.bval
        return FourState(shape=out, aval=aval, bval=bval)
    if op in (UnaryOp.NEG, UnaryOp.RED_AND, UnaryOp.RED_XOR):
        return _all_x(out)
    if op == UnaryOp.BOOL:
        # If any known bit is 1, result is 1; otherwise X
        known_ones = a.aval & ~a.bval
        if known_ones:
            return FourState.from_int(1, out)
        return _all_x(out)
    if op == UnaryOp.RED_OR:
        known_ones = a.aval & ~a.bval
        if known_ones:
            return FourState.from_int(1, out)
        return _all_x(out)
    raise ValueError(f"Unknown unary op: {op}")


def _eval_binary_4(op: BinaryOp, left: FourState, right: FourState, out: Shape) -> FourState:
    # If both fully defined, fast path
    if left.is_fully_defined and right.is_fully_defined:
        return _binary_defined(op, left, right, out)
    return _binary_with_x(op, left, right, out)


def _binary_defined(op: BinaryOp, left: FourState, right: FourState, out: Shape) -> FourState:
    """Both operands are fully defined — use plain integer math."""
    lv = _to_signed(left.aval, left.shape)
    rv = _to_signed(right.aval, right.shape)
    lu = left.aval  # unsigned
    ru = right.aval

    if op == BinaryOp.ADD:
        return FourState.from_int(lv + rv, out)
    if op == BinaryOp.SUB:
        return FourState.from_int(lv - rv, out)
    if op == BinaryOp.MUL:
        return FourState.from_int(lv * rv, out)
    if op == BinaryOp.DIV:
        if rv == 0:
            return _all_x(out)
        return FourState.from_int(int(lv / rv), out)
    if op == BinaryOp.MOD:
        if rv == 0:
            return _all_x(out)
        return FourState.from_int(lv % rv, out)
    if op == BinaryOp.AND:
        return FourState.from_int(lu & ru, out)
    if op == BinaryOp.OR:
        return FourState.from_int(lu | ru, out)
    if op == BinaryOp.XOR:
        return FourState.from_int(lu ^ ru, out)
    if op == BinaryOp.SHL:
        return FourState.from_int(lu << ru, out)
    if op == BinaryOp.SHR:
        if left.shape.signed:
            return FourState.from_int(lv >> rv, out)
        return FourState.from_int(lu >> rv, out)
    if op == BinaryOp.EQ:
        return FourState.from_int(1 if lv == rv else 0, out)
    if op == BinaryOp.NE:
        return FourState.from_int(1 if lv != rv else 0, out)
    if op == BinaryOp.LT:
        return FourState.from_int(1 if lv < rv else 0, out)
    if op == BinaryOp.LE:
        return FourState.from_int(1 if lv <= rv else 0, out)
    if op == BinaryOp.GT:
        return FourState.from_int(1 if lv > rv else 0, out)
    if op == BinaryOp.GE:
        return FourState.from_int(1 if lv >= rv else 0, out)
    if op == BinaryOp.LOGIC_AND:
        return FourState.from_int(1 if lv and rv else 0, out)
    if op == BinaryOp.LOGIC_OR:
        return FourState.from_int(1 if lv or rv else 0, out)
    raise ValueError(f"Unknown binary op: {op}")


def _binary_with_x(op: BinaryOp, left: FourState, right: FourState, out: Shape) -> FourState:
    """At least one operand has X/Z bits."""
    # Bitwise ops: per-bit X propagation
    if op == BinaryOp.AND:
        # 0 & X = 0 (known); 1 & X = X; X & X = X
        #   result aval = la & ra & ~lb & ~rb
        #   result bval = where outcome is indeterminate
        la, lb = left.aval, left.bval
        ra, rb = right.aval, right.bval
        known_zeros_l = ~la & ~lb  # bits that are definitely 0 in left
        known_zeros_r = ~ra & ~rb  # bits that are definitely 0 in right
        known_zero = known_zeros_l | known_zeros_r
        mask = (1 << out.width) - 1
        bval = ~known_zero & mask  # everything else is X
        aval = la & ra & ~bval & mask
        return FourState(shape=out, aval=aval, bval=bval)

    if op == BinaryOp.OR:
        # 1 | X = 1 (known); 0 | X = X; X | X = X
        la, lb = left.aval, left.bval
        ra, rb = right.aval, right.bval
        known_ones_l = la & ~lb
        known_ones_r = ra & ~rb
        known_one = known_ones_l | known_ones_r
        mask = (1 << out.width) - 1
        bval = ~known_one & (lb | rb) & mask
        aval = (known_one | (la & ra & ~bval)) & mask
        return FourState(shape=out, aval=aval, bval=bval)

    if op == BinaryOp.XOR:
        # Any X in → that bit is X
        lb, rb = left.bval, right.bval
        bval = lb | rb
        mask = (1 << out.width) - 1
        bval &= mask
        aval = (left.aval ^ right.aval) & ~bval & mask
        return FourState(shape=out, aval=aval, bval=bval)

    # Logical ops with partial X rules
    if op == BinaryOp.LOGIC_AND:
        l_known_zero = left.is_fully_defined and left.aval == 0
        r_known_zero = right.is_fully_defined and right.aval == 0
        if l_known_zero or r_known_zero:
            return FourState.from_int(0, out)
        l_known_nonzero = left.is_fully_defined and left.aval != 0
        r_known_nonzero = right.is_fully_defined and right.aval != 0
        if l_known_nonzero and r_known_nonzero:
            return FourState.from_int(1, out)
        return _all_x(out)

    if op == BinaryOp.LOGIC_OR:
        l_known_nonzero = left.is_fully_defined and left.aval != 0
        r_known_nonzero = right.is_fully_defined and right.aval != 0
        if l_known_nonzero or r_known_nonzero:
            return FourState.from_int(1, out)
        l_known_zero = left.is_fully_defined and left.aval == 0
        r_known_zero = right.is_fully_defined and right.aval == 0
        if l_known_zero and r_known_zero:
            return FourState.from_int(0, out)
        return _all_x(out)

    # Arithmetic, shift, comparison with X: result is all-X
    return _all_x(out)


def _to_signed(aval: int, shape: Shape) -> int:
    """Interpret unsigned bits as signed if shape says so."""
    if shape.signed and shape.width > 0 and (aval >> (shape.width - 1)) & 1:
        return aval - (1 << shape.width)
    return aval
