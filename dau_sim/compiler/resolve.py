"""Wire resolution for multi-driver nets.

When multiple combinational blocks drive the same signal, we must merge
the driven values according to the net's resolution semantics.

Resolution rules (per IEEE 1800-2017):
  - WIRE / TRI: Z resolves to the non-Z driver; two non-Z drivers → X
  - WAND: driven values ANDed together (Z treated as all-1)
  - WOR: driven values ORed together (Z treated as all-0)
"""

from __future__ import annotations

from dau_sim.ir.types import FourState, NetKind, Shape


def resolve_drivers(
    drivers: list[FourState],
    kind: NetKind,
    shape: Shape,
) -> FourState:
    """Resolve multiple driver values into a single net value.

    Args:
        drivers: One or more FourState values, one per driver.
        kind: The net resolution semantics.
        shape: The shape of the resulting signal.

    Returns:
        Resolved FourState value.
    """
    if not drivers:
        return FourState.z(shape)

    if len(drivers) == 1:
        return drivers[0]

    if kind in (NetKind.WIRE, NetKind.TRI):
        return _resolve_tri(drivers, shape)
    if kind == NetKind.WAND:
        return _resolve_wand(drivers, shape)
    if kind == NetKind.WOR:
        return _resolve_wor(drivers, shape)

    raise ValueError(f"Unknown net kind: {kind}")


def _resolve_tri(drivers: list[FourState], shape: Shape) -> FourState:
    """Tri-state resolution: Z yields to non-Z; conflicting non-Z → X.

    Per-bit truth table (d1, d2):
      0, Z → 0    Z, 0 → 0
      1, Z → 1    Z, 1 → 1
      0, 1 → X    1, 0 → X
      X, * → X    *, X → X
      Z, Z → Z
    """
    mask = (1 << shape.width) - 1 if shape.width else 0
    # Start with all-Z
    result_a = 0
    result_b = mask  # all Z initially

    for drv in drivers:
        da, db = drv.aval & mask, drv.bval & mask
        ra, rb = result_a, result_b

        # Bits where result is currently Z
        rz = ~ra & rb & mask
        # Bits where driver is Z
        dz = ~da & db & mask
        # Bits where driver is X
        dx = da & db & mask
        # Bits where result is X
        rx = ra & rb & mask

        # Z resolves: where result is Z and driver is non-Z, take driver
        take_drv = rz & ~dz & ~dx & mask
        # Where driver is Z, keep current result
        keep_res = dz & mask
        # Both non-Z and different → X
        both_defined = ~rz & ~rx & ~dz & ~dx & mask
        conflict = both_defined & (ra ^ da) & mask
        # Both non-Z and same → keep value
        agree = both_defined & ~(ra ^ da) & mask

        new_a = (ra & keep_res) | (da & take_drv) | (ra & agree) & mask
        new_b = (rb & keep_res & dz) | (conflict) | (rx) | (dx & ~rz) & mask

        result_a = new_a & mask
        result_b = new_b & mask

    return FourState(shape=shape, aval=result_a, bval=result_b)


def _resolve_wand(drivers: list[FourState], shape: Shape) -> FourState:
    """Wired-AND: AND all drivers together. Z treated as all-1."""
    mask = (1 << shape.width) - 1 if shape.width else 0
    result_a = mask  # start with all-1
    result_b = 0

    for drv in drivers:
        da, db = drv.aval & mask, drv.bval & mask
        is_z = ~da & db & mask  # Z bits → treated as 1 (no effect on AND)
        is_x = da & db & mask

        # Non-Z, non-X bits participate in AND
        defined = ~(is_z | is_x) & mask
        result_a = (result_a & (da | ~defined)) & mask
        result_b = (result_b | is_x) & mask

    return FourState(shape=shape, aval=result_a & ~result_b, bval=result_b)


def _resolve_wor(drivers: list[FourState], shape: Shape) -> FourState:
    """Wired-OR: OR all drivers together. Z treated as all-0."""
    mask = (1 << shape.width) - 1 if shape.width else 0
    result_a = 0  # start with all-0
    result_b = 0

    for drv in drivers:
        da, db = drv.aval & mask, drv.bval & mask
        is_z = ~da & db & mask  # Z bits → treated as 0 (no effect on OR)
        is_x = da & db & mask

        defined = ~(is_z | is_x) & mask
        result_a = (result_a | (da & defined)) & mask
        result_b = (result_b | is_x) & mask

    return FourState(shape=shape, aval=result_a & ~result_b, bval=result_b)
