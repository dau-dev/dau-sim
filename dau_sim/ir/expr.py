from enum import Enum, auto

from ccflow import BaseModel
from pydantic import ConfigDict, model_validator

from dau_sim.ir.types import Shape


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


class Expr(BaseModel):
    """Base class for all IR expressions."""

    model_config = ConfigDict(frozen=True)

    shape: Shape


class Const(Expr):
    """Constant integer value.

    Value is stored as a Python int, truncated/sign-extended to the given shape
    when evaluated.
    """

    value: int

    def __repr__(self) -> str:
        return f"{self.shape.width}'{'s' if self.shape.signed else ''}d{self.value}"


class SignalRef(Expr):
    """Reference to a named signal.

    `name` is the signal's unique name within its containing module.
    """

    name: str

    def __repr__(self) -> str:
        return self.name


class Unary(Expr):
    """Unary operation."""

    op: UnaryOp
    operand: Expr


class Binary(Expr):
    """Binary operation."""

    op: BinaryOp
    left: Expr
    right: Expr


class Mux(Expr):
    """Two-input multiplexer (ternary operator).

    sel ? if_true : if_false
    """

    sel: Expr
    if_true: Expr
    if_false: Expr


class Concat(Expr):
    """Bit concatenation.

    parts[0] is MSB, parts[-1] is LSB.
    """

    parts: tuple[Expr, ...]


class Slice(Expr):
    """Bit slice extraction.

    Extracts bits [low:high) from value (0-indexed, exclusive upper bound).
    """

    value: Expr
    low: int
    high: int

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.low < 0 or self.high < self.low:
            raise ValueError(f"Invalid slice bounds: [{self.low}:{self.high})")
        if self.shape.width != self.high - self.low:
            raise ValueError(f"Slice shape width {self.shape.width} doesn't match bounds [{self.low}:{self.high})")
        return self


class SysRandom(Expr):
    """``$random`` system function.

    Returns a pseudo-random 32-bit signed integer.  An optional *seed*
    expression, when provided, seeds the PRNG on first call (matching
    Verilog ``$random(seed)`` semantics).
    """

    seed: Expr | None = None
