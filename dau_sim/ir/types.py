from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PortDirection(Enum):
    """Direction of a module port."""

    INPUT = auto()
    OUTPUT = auto()
    INOUT = auto()


class EdgePolarity(Enum):
    """Clock edge sensitivity."""

    POSEDGE = auto()
    NEGEDGE = auto()
    BOTH = auto()


class ResetStyle(Enum):
    """Reset behavior."""

    SYNC = auto()
    ASYNC = auto()


class NetKind(Enum):
    """Net resolution semantics for multi-driver signals."""

    WIRE = auto()  # default; multiple drivers are an error unless resolved
    TRI = auto()  # tri-state: Z resolves to the non-Z driver
    WAND = auto()  # wired-AND: drivers ANDed together
    WOR = auto()  # wired-OR: drivers ORed together


@dataclass(frozen=True)
class Shape:
    """Bit-width and signedness of a signal.

    Canonical type representation shared across all frontends.
    """

    width: int
    signed: bool = False

    def __post_init__(self):
        if self.width < 0:
            raise ValueError(f"Shape width must be non-negative, got {self.width}")

    @property
    def max_value(self) -> int:
        if self.signed:
            return (1 << (self.width - 1)) - 1
        return (1 << self.width) - 1

    @property
    def min_value(self) -> int:
        if self.signed:
            return -(1 << (self.width - 1))
        return 0

    def __repr__(self) -> str:
        s = "signed" if self.signed else "unsigned"
        return f"Shape({self.width}, {s})"


@dataclass(frozen=True)
class FourState:
    """Four-state value using the VPI convention.

    Two bitmaps of ``width`` bits:
      aval=0, bval=0 → logic 0
      aval=1, bval=0 → logic 1
      aval=0, bval=1 → Z (high-impedance)
      aval=1, bval=1 → X (unknown)

    ``shape`` determines the width and signedness.
    """

    shape: Shape
    aval: int  # value bits
    bval: int  # mask bits (non-zero → X or Z)

    def __post_init__(self) -> None:
        mask = (1 << self.shape.width) - 1 if self.shape.width else 0
        # Normalize to width
        object.__setattr__(self, "aval", self.aval & mask)
        object.__setattr__(self, "bval", self.bval & mask)

    @property
    def has_unknown(self) -> bool:
        """True if any bit is X or Z."""
        return self.bval != 0

    @property
    def is_fully_defined(self) -> bool:
        """True if all bits are 0 or 1 (no X/Z)."""
        return self.bval == 0

    @property
    def to_int(self) -> int | None:
        """Convert to Python int.  Returns None if any bits are X/Z."""
        if self.has_unknown:
            return None
        return _signed_int(self.aval, self.shape)

    @classmethod
    def from_int(cls, value: int, shape: Shape) -> FourState:
        """Create a fully-defined FourState value from a Python int."""
        mask = (1 << shape.width) - 1 if shape.width else 0
        return cls(shape=shape, aval=value & mask, bval=0)

    @classmethod
    def x(cls, shape: Shape) -> FourState:
        """Create an all-X value of the given shape."""
        mask = (1 << shape.width) - 1 if shape.width else 0
        return cls(shape=shape, aval=mask, bval=mask)

    @classmethod
    def z(cls, shape: Shape) -> FourState:
        """Create an all-Z value of the given shape."""
        mask = (1 << shape.width) - 1 if shape.width else 0
        return cls(shape=shape, aval=0, bval=mask)

    def __repr__(self) -> str:
        if self.is_fully_defined:
            v = self.to_int
            return f"{self.shape.width}'d{v}"
        # Show per-bit: 0/1/x/z
        bits = []
        for i in range(self.shape.width - 1, -1, -1):
            a = (self.aval >> i) & 1
            b = (self.bval >> i) & 1
            if b == 0:
                bits.append(str(a))
            elif a == 1:
                bits.append("x")
            else:
                bits.append("z")
        return "".join(bits)


def _signed_int(raw: int, shape: Shape) -> int:
    """Convert an unsigned bit-pattern to a signed int if shape.signed."""
    if shape.signed and shape.width > 0 and (raw >> (shape.width - 1)) & 1:
        return raw - (1 << shape.width)
    return raw
