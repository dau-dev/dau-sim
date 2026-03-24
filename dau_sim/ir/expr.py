from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from dau_sim.ir.types import Shape

if TYPE_CHECKING:
    pass


class UnaryOp(Enum):
    """Unary operators."""

    NOT = auto()  # bitwise ~
    NEG = auto()  # arithmetic -
    BOOL = auto()  # logical bool (reduction OR, != 0)
    # Reduction operators
    RED_AND = auto()  # &x
    RED_OR = auto()  # |x
    RED_XOR = auto()  # ^x


class BinaryOp(Enum):
    """Binary operators."""

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()  # integer division, truncates toward zero
    MOD = auto()

    # Bitwise
    AND = auto()
    OR = auto()
    XOR = auto()

    # Shift
    SHL = auto()
    SHR = auto()  # logical shift (unsigned) or arithmetic shift (signed)

    # Comparison (result is always 1-bit unsigned)
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()

    # Logical
    LOGIC_AND = auto()
    LOGIC_OR = auto()


@dataclass(frozen=True)
class Expr:
    """Base class for all IR expressions."""

    shape: Shape


@dataclass(frozen=True)
class Const(Expr):
    """Constant integer value.

    Value is stored as a Python int, truncated/sign-extended to the given shape
    when evaluated.
    """

    value: int

    def __repr__(self) -> str:
        return f"{self.shape.width}'{'s' if self.shape.signed else ''}d{self.value}"


@dataclass(frozen=True)
class SignalRef(Expr):
    """Reference to a named signal.

    `name` is the signal's unique name within its containing module.
    """

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Unary(Expr):
    """Unary operation."""

    op: UnaryOp
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    """Binary operation."""

    op: BinaryOp
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Mux(Expr):
    """Two-input multiplexer (ternary operator).

    sel ? if_true : if_false
    """

    sel: Expr
    if_true: Expr
    if_false: Expr


@dataclass(frozen=True)
class Concat(Expr):
    """Bit concatenation.

    parts[0] is MSB, parts[-1] is LSB.
    """

    parts: tuple[Expr, ...]


@dataclass(frozen=True)
class Slice(Expr):
    """Bit slice extraction.

    Extracts bits [low:high) from value (0-indexed, exclusive upper bound).
    """

    value: Expr
    low: int
    high: int

    def __post_init__(self):
        if self.low < 0 or self.high < self.low:
            raise ValueError(f"Invalid slice bounds: [{self.low}:{self.high})")
        if self.shape.width != self.high - self.low:
            raise ValueError(f"Slice shape width {self.shape.width} doesn't match bounds [{self.low}:{self.high})")
