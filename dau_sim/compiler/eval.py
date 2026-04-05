"""Bit-accurate expression evaluator.

Evaluates IR expression trees with correct bit-width semantics.
Values are Python ints, masked/truncated to the signal's width.
"""

from __future__ import annotations

import random as _random

from dau_sim.ir.expr import (
    Binary,
    BinaryOp,
    Concat,
    Const,
    Expr,
    Mux,
    SignalRef,
    Slice,
    SysRandom,
    Unary,
    UnaryOp,
)
from dau_sim.ir.types import Shape

# Module-level PRNG instance for $random — seeded lazily.
_sys_random_rng: _random.Random = _random.Random()
_WIDTH_MASKS: dict[int, int] = {}


def _width_mask(width: int) -> int:
    """Return a cached (1 << width) - 1 bitmask for non-negative widths."""
    if width <= 0:
        return 0
    cached = _WIDTH_MASKS.get(width)
    if cached is None:
        cached = (1 << width) - 1
        _WIDTH_MASKS[width] = cached
    return cached


def mask_value(value: int, shape: Shape) -> int:
    """Truncate/sign-extend value to fit in shape."""
    width = shape.width
    if width == 0:
        return 0
    mask = _width_mask(width)
    raw = value & mask
    if shape.signed and ((raw >> (width - 1)) & 1):
        raw -= 1 << width
    return raw


def eval_expr(expr: Expr, signals: dict[str, int]) -> int:
    """Evaluate an expression tree given current signal values.

    Returns an int value truncated to the expression's shape.
    """
    expr_type = type(expr)

    if expr_type is Const:
        width = expr.shape.width
        if width == 0:
            return 0
        mask = _width_mask(width)
        raw = expr.value & mask
        if expr.shape.signed and ((raw >> (width - 1)) & 1):
            raw -= 1 << width
        return raw

    if expr_type is SignalRef:
        width = expr.shape.width
        if width == 0:
            return 0
        mask = _width_mask(width)
        raw = signals[expr.name] & mask
        if expr.shape.signed and ((raw >> (width - 1)) & 1):
            raw -= 1 << width
        return raw

    if expr_type is Unary:
        a = eval_expr(expr.operand, signals)
        return _eval_unary(expr.op, a, expr.operand.shape, expr.shape)

    if expr_type is Binary:
        left = eval_expr(expr.left, signals)
        right = eval_expr(expr.right, signals)
        return _eval_binary(expr.op, left, right, expr.left.shape, expr.right.shape, expr.shape)

    if expr_type is Mux:
        sel = eval_expr(expr.sel, signals)
        if sel:
            return mask_value(eval_expr(expr.if_true, signals), expr.shape)
        return mask_value(eval_expr(expr.if_false, signals), expr.shape)

    if expr_type is Concat:
        result = 0
        for part in expr.parts:
            result = (result << part.shape.width) | (eval_expr(part, signals) & _width_mask(part.shape.width))
        return mask_value(result, expr.shape)

    if expr_type is Slice:
        val = eval_expr(expr.value, signals)
        # Extract bits [low:high)
        extracted = (val >> expr.low) & _width_mask(expr.high - expr.low)
        return mask_value(extracted, expr.shape)

    if expr_type is SysRandom:
        if expr.seed is not None:
            seed_val = eval_expr(expr.seed, signals)
            _sys_random_rng.seed(seed_val)
        # Verilog $random returns a 32-bit signed integer
        return mask_value(_sys_random_rng.randint(-(1 << 31), (1 << 31) - 1), expr.shape)

    raise TypeError(f"Unknown expression type: {type(expr).__name__}")


def _eval_unary(op: UnaryOp, a: int, a_shape: Shape, out_shape: Shape) -> int:
    if op is UnaryOp.NOT:
        return mask_value(~a, out_shape)
    if op is UnaryOp.NEG:
        return mask_value(-a, out_shape)
    if op is UnaryOp.BOOL:
        return 1 if a != 0 else 0
    if op is UnaryOp.RED_AND:
        all_ones = _width_mask(a_shape.width)
        return 1 if (a & all_ones) == all_ones else 0
    if op is UnaryOp.RED_OR:
        return 1 if a != 0 else 0
    if op is UnaryOp.RED_XOR:
        # Parity: count set bits in width
        val = a & _width_mask(a_shape.width)
        count = val.bit_count()
        return count & 1
    raise ValueError(f"Unknown unary op: {op}")


def _eval_binary(
    op: BinaryOp,
    left: int,
    right: int,
    l_shape: Shape,
    r_shape: Shape,
    out_shape: Shape,
) -> int:
    if op is BinaryOp.ADD:
        return mask_value(left + right, out_shape)
    if op is BinaryOp.SUB:
        return mask_value(left - right, out_shape)
    if op is BinaryOp.MUL:
        return mask_value(left * right, out_shape)
    if op is BinaryOp.DIV:
        if right == 0:
            return 0  # X in real hardware; 0 is safe default
        # Truncation toward zero for signed
        if l_shape.signed or r_shape.signed:
            return mask_value(int(left / right), out_shape)
        return mask_value(left // right, out_shape)
    if op is BinaryOp.MOD:
        if right == 0:
            return 0
        return mask_value(left % right, out_shape)
    if op is BinaryOp.AND:
        return mask_value(left & right, out_shape)
    if op is BinaryOp.OR:
        return mask_value(left | right, out_shape)
    if op is BinaryOp.XOR:
        return mask_value(left ^ right, out_shape)
    if op is BinaryOp.SHL:
        return mask_value(left << right, out_shape)
    if op is BinaryOp.SHR:
        if l_shape.signed:
            return mask_value(left >> right, out_shape)
        # Unsigned: ensure no sign extension
        unsigned_left = left & _width_mask(l_shape.width)
        return mask_value(unsigned_left >> right, out_shape)
    # Comparison operators — always produce 1-bit result
    if op is BinaryOp.EQ:
        return 1 if left == right else 0
    if op is BinaryOp.NE:
        return 1 if left != right else 0
    if op is BinaryOp.LT:
        return 1 if left < right else 0
    if op is BinaryOp.LE:
        return 1 if left <= right else 0
    if op is BinaryOp.GT:
        return 1 if left > right else 0
    if op is BinaryOp.GE:
        return 1 if left >= right else 0
    if op is BinaryOp.LOGIC_AND:
        return 1 if (left != 0 and right != 0) else 0
    if op is BinaryOp.LOGIC_OR:
        return 1 if (left != 0 or right != 0) else 0
    raise ValueError(f"Unknown binary op: {op}")
