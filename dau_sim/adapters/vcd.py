"""VCD (Value Change Dump) writer for dau-sim simulation traces.

Produces IEEE 1364-2001 compliant VCD files from the simulation trace
format (``dict[str, list[tuple[datetime, int]]]``) produced by
``CompiledModule.run()``.

Usage::

    from dau_sim.adapters.vcd import write_vcd, traces_to_vcd

    # From a CompiledModule run:
    traces = compiled.run(cycles=100, inputs={"d": 1})
    write_vcd("out.vcd", traces, module=compiled.module)

    # Or get VCD as a string:
    vcd_str = traces_to_vcd(traces, module=compiled.module)
"""

from __future__ import annotations

import io
import string
from datetime import datetime
from pathlib import Path
from typing import TextIO

from dau_sim.ir.module import Module
from dau_sim.ir.types import EdgePolarity, Shape

_ID_CHARS = string.printable[:-5]  # printable minus whitespace

__all__ = (
    "traces_to_vcd",
    "write_vcd",
)


def _make_id(index: int) -> str:
    """Generate a short VCD identifier code from an integer index.

    VCD identifiers are one or more printable ASCII characters (33–126).
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    result = []
    base = len(_ID_CHARS)
    while True:
        result.append(_ID_CHARS[index % base])
        index = index // base
        if index == 0:
            break
        index -= 1  # shift so 0→first char, not 1→first char on next digit
    return "".join(result)


def _datetime_to_timescale_ticks(
    timestamps: list[datetime],
    timescale_ns: int = 1,
) -> list[int]:
    """Convert datetime timestamps to integer VCD time ticks.

    All timestamps are relative to the first timestamp in the list.
    """
    if not timestamps:
        return []
    base = timestamps[0]
    ticks = []
    for ts in timestamps:
        delta = ts - base
        ns = int(delta.total_seconds() * 1_000_000_000)
        ticks.append(ns // timescale_ns)
    return ticks


def _format_value(value: int, width: int) -> str:
    """Format an integer value as a VCD binary literal."""
    if width == 1:
        return str(value & 1)
    # Multi-bit: binary string, MSB first, padded to width
    if value < 0:
        # Two's complement representation for signed values
        value = value & ((1 << width) - 1)
    bits = bin(value)[2:]  # strip '0b'
    return "b" + bits.zfill(width)


def _write_header(
    out: TextIO,
    module_name: str,
    signals: dict[str, tuple[str, int, str]],
    timescale: str = "1ns",
    date: str | None = None,
) -> None:
    """Write the VCD header section.

    Parameters
    ----------
    out : TextIO
        Output stream.
    module_name : str
        Top-level scope name.
    signals : dict
        Maps signal name → (vcd_id, width, var_type).
        var_type is "wire", "reg", etc.
    timescale : str
        VCD timescale string (e.g. "1ns", "1ps").
    date : str | None
        Date string for $date section.
    """
    if date:
        out.write(f"$date\n  {date}\n$end\n")
    out.write("$version dau-sim 0.1.0 $end\n")
    out.write(f"$timescale {timescale} $end\n")

    # Scope
    out.write(f"$scope module {module_name} $end\n")
    for sig_name, (vcd_id, width, var_type) in signals.items():
        out.write(f"$var {var_type} {width} {vcd_id} {sig_name} $end\n")
    out.write("$upscope $end\n")

    out.write("$enddefinitions $end\n")


def _write_initial_values(
    out: TextIO,
    signals: dict[str, tuple[str, int, str]],
    initial_values: dict[str, int],
) -> None:
    """Write the $dumpvars section with initial values."""
    out.write("$dumpvars\n")
    for sig_name, (vcd_id, width, _) in signals.items():
        val = initial_values.get(sig_name, 0)
        formatted = _format_value(val, width)
        if width == 1:
            out.write(f"{formatted}{vcd_id}\n")
        else:
            out.write(f"{formatted} {vcd_id}\n")
    out.write("$end\n")


def _write_changes(
    out: TextIO,
    signals: dict[str, tuple[str, int, str]],
    traces: dict[str, list[tuple[datetime, int]]],
    timescale_ns: int = 1,
    base_time: datetime | None = None,
) -> None:
    """Write time-stamped value changes."""
    # Collect all events across all signals, sorted by time
    events: list[tuple[int, str, int]] = []  # (tick, signal_name, value)

    # Find the global base time if not provided
    if base_time is None:
        for sig_name, trace in traces.items():
            if trace and (base_time is None or trace[0][0] < base_time):
                base_time = trace[0][0]

    if base_time is None:
        return

    for sig_name, trace in traces.items():
        if sig_name not in signals:
            continue
        for ts, val in trace:
            delta = ts - base_time
            ns = int(delta.total_seconds() * 1_000_000_000)
            tick = ns // timescale_ns
            events.append((tick, sig_name, val))

    # Sort by tick, then by signal name for determinism
    events.sort(key=lambda e: (e[0], e[1]))

    current_tick = -1
    for tick, sig_name, value in events:
        if tick != current_tick:
            out.write(f"#{tick}\n")
            current_tick = tick
        vcd_id, width, _ = signals[sig_name]
        formatted = _format_value(value, width)
        if width == 1:
            out.write(f"{formatted}{vcd_id}\n")
        else:
            out.write(f"{formatted} {vcd_id}\n")


def traces_to_vcd(
    traces: dict[str, list[tuple[datetime, int]]],
    *,
    module: Module | None = None,
    timescale: str = "1ns",
    scope: str | None = None,
) -> str:
    """Convert simulation traces to a VCD string.

    Parameters
    ----------
    traces : dict
        Signal traces from ``CompiledModule.run()``.
    module : Module | None
        IR module (used for signal widths and port directions).
        If None, all signals are assumed 1-bit wire.
    timescale : str
        VCD timescale (e.g. "1ns", "1ps", "10ns").
    scope : str | None
        Top-level scope name. Defaults to module name or "top".

    Returns
    -------
    str
        Complete VCD file content.
    """
    buf = io.StringIO()
    _traces_to_vcd_stream(buf, traces, module=module, timescale=timescale, scope=scope)
    return buf.getvalue()


def write_vcd(
    path: str | Path,
    traces: dict[str, list[tuple[datetime, int]]],
    *,
    module: Module | None = None,
    timescale: str = "1ns",
    scope: str | None = None,
) -> None:
    """Write simulation traces to a VCD file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    traces : dict
        Signal traces from ``CompiledModule.run()``.
    module : Module | None
        IR module for signal metadata.
    timescale : str
        VCD timescale string.
    scope : str | None
        Top-level scope name.
    """
    with open(path, "w") as f:
        _traces_to_vcd_stream(f, traces, module=module, timescale=timescale, scope=scope)


def _traces_to_vcd_stream(
    out: TextIO,
    traces: dict[str, list[tuple[datetime, int]]],
    *,
    module: Module | None = None,
    timescale: str = "1ns",
    scope: str | None = None,
) -> None:
    """Core implementation: write VCD to a stream."""
    # Parse timescale to get ns multiplier
    timescale_ns = _parse_timescale_ns(timescale)

    # Build signal metadata
    shapes: dict[str, Shape] = {}
    var_types: dict[str, str] = {}

    if module is not None:
        scope_name = scope or module.name
        for p in module.ports:
            shapes[p.name] = p.shape
            var_types[p.name] = "wire"
        for s in module.signals:
            shapes[s.name] = s.shape
            var_types[s.name] = "reg"
    else:
        scope_name = scope or "top"

    # Build VCD signal table: name → (id, width, type)
    signal_table: dict[str, tuple[str, int, str]] = {}
    idx = 0
    for sig_name in sorted(traces.keys()):
        shape = shapes.get(sig_name, Shape(1, False))
        vtype = var_types.get(sig_name, "wire")
        signal_table[sig_name] = (_make_id(idx), shape.width, vtype)
        idx += 1

    # Write header
    _write_header(out, scope_name, signal_table, timescale=timescale)

    # Clock synthesis setup
    # Detect clock signals and their active/inactive values from the module.
    # The simulation engine only emits trace data on active clock edges,
    # so the VCD writer must synthesize the inactive (opposite) edges to
    # produce a proper square-wave clock in the output.
    clock_info: dict[str, tuple[int, int]] = {}  # {clk_name: (active, inactive)}
    if module is not None:
        for cd in module.clock_domains:
            if cd.clk in signal_table:
                if cd.edge == EdgePolarity.POSEDGE:
                    clock_info[cd.clk] = (1, 0)
                else:
                    clock_info[cd.clk] = (0, 1)

    # Infer half-period from the gap between consecutive trace timestamps.
    half_period_td = None
    if clock_info:
        for trace in traces.values():
            if len(trace) >= 2:
                half_period_td = (trace[1][0] - trace[0][0]) / 2
                break

    synthesize_clocks = bool(clock_info and half_period_td is not None)

    # Initial values
    initial_values: dict[str, int] = {}
    if synthesize_clocks:
        # With clock synthesis the VCD timeline is shifted back by one
        # half-period so that $dumpvars represents the state *before* the
        # first active edge.  Clocks start at their inactive level and
        # non-clock signals start at zero (reset state).
        for sig_name in signal_table:
            if sig_name in clock_info:
                initial_values[sig_name] = clock_info[sig_name][1]
            else:
                initial_values[sig_name] = 0
    else:
        for sig_name, trace in traces.items():
            initial_values[sig_name] = trace[0][1] if trace else 0

    # Write initial values
    _write_initial_values(out, signal_table, initial_values)

    # Compute the base time from the full (untrimmed) traces
    base_time: datetime | None = None
    for trace in traces.values():
        if trace and (base_time is None or trace[0][0] < base_time):
            base_time = trace[0][0]

    if synthesize_clocks and base_time is not None:
        base_time = base_time - half_period_td

    # Build value-change traces
    trimmed: dict[str, list[tuple[datetime, int]]] = {}
    if synthesize_clocks:
        # Include ALL entries for non-clock signals (dumpvars has zeros,
        # so the first trace entry is a real change at the first posedge).
        for sig_name, trace in traces.items():
            if sig_name in clock_info:
                # Synthesize clock toggle events
                active_val, inactive_val = clock_info[sig_name]
                events: list[tuple[datetime, int]] = []
                for i, (ts, _) in enumerate(trace):
                    events.append((ts, active_val))
                    if i < len(trace) - 1:
                        mid = ts + (trace[i + 1][0] - ts) / 2
                        events.append((mid, inactive_val))
                    elif half_period_td:
                        events.append((ts + half_period_td, inactive_val))
                trimmed[sig_name] = events
            else:
                trimmed[sig_name] = list(trace)
    else:
        # Original behaviour: first entry is in $dumpvars, changes start
        # from the second entry onward.
        for sig_name, trace in traces.items():
            trimmed[sig_name] = trace[1:] if len(trace) > 1 else []

    _write_changes(out, signal_table, trimmed, timescale_ns=timescale_ns, base_time=base_time)


def _parse_timescale_ns(timescale: str) -> int:
    """Parse a VCD timescale string to nanoseconds.

    Supports: "1ns", "10ns", "100ns", "1us", "1ps", etc.
    """
    ts = timescale.strip().lower()
    units = {"ps": 0.001, "ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000}
    for suffix, mult in units.items():
        if ts.endswith(suffix):
            num = int(ts[: -len(suffix)])
            return max(1, int(num * mult))
    return 1  # default to 1ns
