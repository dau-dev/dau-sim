"""cocotb backend for dau-sim.

Provides a pure-Python implementation of cocotb's ``simulator`` module (the GPI
layer) backed by dau-sim's IR evaluation engine.  This allows existing cocotb
testbenches to run on dau-sim **without any intermediary compilation step**.

Usage::

    from dau_sim.backends.cocotb_backend import run_cocotb

    # With an Amaranth design:
    run_cocotb(my_design, test_module="test_my_design")

    # With an IR Module:
    run_cocotb(ir_module, test_module="test_my_design")
"""

from __future__ import annotations

import heapq
import logging
import os
import random
import sys
import time
import types
from typing import Any, Callable

from dau_sim.compiler.compile import (
    _build_domain_info,
    _collect_init_values,
    _edge_fires,
    _exec_mem_reads,
    _exec_mem_writes,
    _exec_stmts,
)
from dau_sim.compiler.depanalysis import build_assignments, topological_sort
from dau_sim.compiler.eval import mask_value
from dau_sim.ir.module import Module
from dau_sim.ir.types import NetKind, ResetStyle, Shape

__all__ = ("run_cocotb",)

_log = logging.getLogger(__name__)

# NOTE: GPI constants — must match cocotb.simulator
# Signal types (must match cocotb.simulator C extension)
MODULE = 2
LOGIC = 15
LOGIC_ARRAY = 16
INTEGER = 10
REAL = 9
ENUM = 7
STRING = 11
GENARRAY = 12
STRUCTURE = 8
NETARRAY = 6
MEMORY = 1
PACKED_STRUCTURE = 14
UNKNOWN = 0
PACKAGE = 13

# Discovery modes
OBJECTS = 1
DRIVERS = 2
LOADS = 3

# Edge types
RISING = 0
FALLING = 1
VALUE_CHANGE = 2

# Range directions
RANGE_UP = 1
RANGE_DOWN = -1
RANGE_NO_DIR = 0


class DauSimCallback:
    """A registered callback handle.  Supports ``deregister()``."""

    def __init__(self, engine: SimulationEngine, cb_id: int):
        self._engine = engine
        self._id = cb_id
        self._active = True

    def deregister(self) -> None:
        if self._active:
            self._engine._deregister_callback(self._id)
            self._active = False


# GPISetAction values (from cocotb C source)
_GPI_DEPOSIT = 0
_GPI_FORCE = 1
_GPI_RELEASE = 2
_GPI_DEPOSIT_IMMEDIATE = 3


class DauSimHandle:
    """Pure-Python GPI handle wrapping an IR signal or module scope.

    Cocotb's ``handle.py`` calls methods on ``gpi_sim_hdl`` objects; this
    class provides the same interface backed by dau-sim's signal state dict.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        name: str,
        path: str | None = None,
        *,
        is_scope: bool = False,
        shape: Shape | None = None,
        children: dict[str, DauSimHandle] | None = None,
        is_const: bool = False,
    ):
        self._engine = engine
        self._name = name
        self._path = path or name
        self._is_scope = is_scope
        self._shape = shape
        self._children: dict[str, DauSimHandle] = children or {}
        self._is_const = is_const
        # For bit-indexed sub-handles (set externally)
        self._parent_signal: str | None = None
        self._bit_index: int = 0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def get_name_string(self) -> str:
        return self._name

    def get_type(self) -> int:
        if self._is_scope:
            return MODULE
        if self._shape is not None:
            return LOGIC if self._shape.width == 1 else LOGIC_ARRAY
        return UNKNOWN

    def get_type_string(self) -> str:
        t = self.get_type()
        _names = {
            MODULE: "GPI_MODULE",
            LOGIC: "GPI_LOGIC",
            LOGIC_ARRAY: "GPI_LOGIC_ARRAY",
            INTEGER: "GPI_INTEGER",
            UNKNOWN: "GPI_UNKNOWN",
        }
        return _names.get(t, "GPI_UNKNOWN")

    def get_definition_name(self) -> str:
        return self._name

    def get_definition_file(self) -> str:
        return ""

    def get_const(self) -> bool:
        return self._is_const

    def get_indexable(self) -> bool:
        return self._shape is not None and self._shape.width > 1

    def get_num_elems(self) -> int:
        if self._shape is not None:
            return self._shape.width
        return len(self._children)

    def get_range(self) -> tuple[int, int, int]:
        if self._shape is not None and self._shape.width > 1:
            return (self._shape.width - 1, 0, RANGE_DOWN)
        return (0, 0, RANGE_NO_DIR)

    def get_handle_by_name(self, name: str, discovery_method: Any = None) -> DauSimHandle | None:  # noqa: ARG002 (cocotb simulator API)
        return self._children.get(name)

    def get_handle_by_index(self, index: int) -> DauSimHandle:
        # For bit-indexing into a logic array, create a single-bit sub-handle
        if self._shape is not None and self._shape.width > 1:
            bit_name = f"{self._name}[{index}]"
            if bit_name not in self._children:
                bit_handle = DauSimHandle(
                    self._engine,
                    bit_name,
                    f"{self._path}[{index}]",
                    shape=Shape(1),
                )
                bit_handle._parent_signal = self._name
                bit_handle._bit_index = index
                self._children[bit_name] = bit_handle
            return self._children[bit_name]
        raise IndexError(f"Cannot index {self._name}")

    def iterate(self, mode: int) -> DauSimIterator:  # noqa: ARG002 (cocotb simulator API)
        return DauSimIterator(list(self._children.values()))

    def get_signal_val_long(self) -> int:
        if self._parent_signal is not None:
            parent_val = self._engine.get_signal(self._parent_signal)
            return (parent_val >> self._bit_index) & 1
        return self._engine.get_signal(self._name)

    def get_signal_val_binstr(self) -> str:
        val = self.get_signal_val_long()
        width = self._shape.width if self._shape else 1
        return format(val & ((1 << width) - 1), f"0{width}b")

    def get_signal_val_real(self) -> float:
        return float(self.get_signal_val_long())

    def get_signal_val_str(self) -> bytes:
        return str(self.get_signal_val_long()).encode()

    def set_signal_val_int(self, action: int, value: int) -> None:  # noqa: ARG002 (cocotb simulator API)
        if self._parent_signal is not None:
            parent_val = self._engine.get_signal(self._parent_signal)
            bit = self._bit_index
            if value & 1:
                parent_val |= 1 << bit
            else:
                parent_val &= ~(1 << bit)
            self._engine.set_signal(self._parent_signal, parent_val)
        else:
            self._engine.set_signal(self._name, value)

    def set_signal_val_binstr(self, action: int, value: str) -> None:
        self.set_signal_val_int(action, int(value, 2))

    def set_signal_val_real(self, action: int, value: float) -> None:
        self.set_signal_val_int(action, int(value))

    def set_signal_val_str(self, action: int, value: bytes) -> None:
        self.set_signal_val_int(action, int(value))


class DauSimIterator:
    """Iterator over child handles."""

    def __init__(self, handles: list[DauSimHandle]):
        self._handles = handles
        self._index = 0

    def __iter__(self) -> DauSimIterator:
        return self

    def __next__(self) -> DauSimHandle:
        if self._index >= len(self._handles):
            raise StopIteration
        h = self._handles[self._index]
        self._index += 1
        return h


# SimulationEngine — time-stepping core
# Callback phases (lower = executed earlier in a timestep)
_PHASE_NORMAL = 0
_PHASE_READWRITE = 1
_PHASE_READONLY = 2
_PHASE_NEXTSTEP = 3


class SimulationEngine:
    """Synchronous simulation engine that drives cocotb's callback scheduler.

    Unlike the CSP-based compiler engine, this engine does **not** auto-toggle
    clock signals.  Clocks are driven externally by cocotb (e.g. via
    ``cocotb.clock.Clock``), and the engine reactively detects clock edges
    from signal value changes, then evaluates sequential logic.

    Callback queue entries: ``(sim_time, phase, seq_id, callback, args, cb_id)``

    Per-timestep execution order:

    1. Advance ``sim_time`` to the earliest queued callback.
    2. Fire **Normal** (timed) callbacks — cocotb tests/clock tasks resume.
    3. Fire **ReadWrite** callbacks — ``_apply_scheduled_writes`` applies
       deferred signal writes from the cocotb write scheduler.
    4. **Settle** — detect clock edges (comparing ``_prev_signals``), execute
       sequential blocks on active edges, handle resets, settle combinational.
       Sequential block outputs are **staged** (NBA semantics) and not yet
       visible to cocotb callbacks.
    5. Fire **value-change callbacks** (``RisingEdge``, ``FallingEdge``, …) —
       cocotb test coroutines see **pre-NBA** signal values, matching real
       HDL simulator behavior.
    5b. **Apply NBA** — sequential block outputs take effect, combinational
        logic re-settles, and any resulting value-change callbacks fire.
    6. If new ReadWrite callbacks were registered during VCH processing,
       repeat from step 3 (delta cycle, combinational only).
    7. Fire **ReadOnly** callbacks.
    8. Snapshot ``_prev_signals`` for the next step.
    """

    def __init__(self, module: Module, time_precision: int = -12):
        self._module = module
        self._time_precision = time_precision  # -12 = 1ps
        self._sim_time: int = 0
        self._running = False
        self._seq_counter = 0

        # IR compilation
        # Flatten if needed
        if module.instances or module.submodules:
            from dau_sim.compiler.flatten import flatten_module

            module = flatten_module(module)
            self._module = module

        # Build shapes
        self._shapes: dict[str, Shape] = {}
        self._net_kinds: dict[str, NetKind] = {}
        for p in module.ports:
            self._shapes[p.name] = p.shape
            self._net_kinds[p.name] = p.signal.net_kind
        for s in module.signals:
            self._shapes[s.name] = s.shape
            self._net_kinds[s.name] = s.net_kind

        # Build comb order
        stmts_list = [(i, cb.stmts) for i, cb in enumerate(module.comb_blocks)]
        assignments = build_assignments(stmts_list)
        self._comb_order = topological_sort(assignments)

        # Domain info
        self._domain_info = _build_domain_info(module)
        self._init_values = _collect_init_values(module)
        self._has_seq = (
            len(module.seq_blocks) > 0
            or any(mem.write_ports for mem in module.memories)
            or any(rp.domain is not None for mem in module.memories for rp in mem.read_ports)
        )

        # Memory state
        self._mem_state: dict[str, list[int]] = {}
        for mem in module.memories:
            init_data = list(mem.init) if mem.init else []
            init_data.extend([0] * (mem.depth - len(init_data)))
            self._mem_state[mem.name] = init_data
            # Register memory port signals in shapes
            for rp in mem.read_ports:
                if rp.addr not in self._shapes:
                    self._shapes[rp.addr] = Shape(max(1, (mem.depth - 1).bit_length()))
                if rp.data not in self._shapes:
                    self._shapes[rp.data] = mem.shape
                if rp.en and rp.en not in self._shapes:
                    self._shapes[rp.en] = Shape(1)
            for wp in mem.write_ports:
                if wp.addr not in self._shapes:
                    self._shapes[wp.addr] = Shape(max(1, (mem.depth - 1).bit_length()))
                if wp.data not in self._shapes:
                    self._shapes[wp.data] = mem.shape
                if wp.en not in self._shapes:
                    en_width = mem.shape.width // wp.granularity if wp.granularity > 0 else 1
                    self._shapes[wp.en] = Shape(en_width)

        # Initialize signal state
        self._signals: dict[str, int] = {}
        for p in module.ports:
            self._signals[p.name] = p.signal.init
        for s in module.signals:
            self._signals[s.name] = s.init

        # Run init blocks
        for ib in module.init_blocks:
            _exec_stmts(ib.stmts, self._signals, self._shapes)

        # Previous signal values (for edge detection and VCH)
        self._prev_signals: dict[str, int] = dict(self._signals)

        # Callback infrastructure
        # Heap of (sim_time, phase, seq_id, callback, args, cb_id)
        self._callback_queue: list[tuple[int, int, int, Callable, tuple, int]] = []
        # Value-change callbacks: signal_name → list of (edge_type, callback, args, cb_id)
        self._vch_callbacks: dict[str, list[tuple[int, Callable, tuple, int]]] = {}
        # Active callback IDs
        self._active_callbacks: set[int] = set()
        self._next_cb_id = 0

        # Handle hierarchy
        self._root_handle: DauSimHandle | None = None

    def _alloc_cb_id(self) -> int:
        cb_id = self._next_cb_id
        self._next_cb_id += 1
        self._active_callbacks.add(cb_id)
        return cb_id

    def _deregister_callback(self, cb_id: int) -> None:
        self._active_callbacks.discard(cb_id)
        # Also remove from VCH callbacks
        for sig_name in list(self._vch_callbacks):
            self._vch_callbacks[sig_name] = [entry for entry in self._vch_callbacks[sig_name] if entry[3] != cb_id]
            if not self._vch_callbacks[sig_name]:
                del self._vch_callbacks[sig_name]

    # Signal access (used by DauSimHandle)
    def get_signal(self, name: str) -> int:
        return self._signals.get(name, 0)

    def set_signal(self, name: str, value: int) -> None:
        if name in self._shapes:
            value = mask_value(value, self._shapes[name])
        self._signals[name] = value

    # Callback registration (simulator API)
    def register_timed_callback(self, time_steps: int, func: Callable, *args: Any) -> DauSimCallback:
        cb_id = self._alloc_cb_id()
        target_time = self._sim_time + time_steps
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._callback_queue, (target_time, _PHASE_NORMAL, seq, func, args, cb_id))
        return DauSimCallback(self, cb_id)

    def register_readonly_callback(self, func: Callable, *args: Any) -> DauSimCallback:
        cb_id = self._alloc_cb_id()
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._callback_queue, (self._sim_time, _PHASE_READONLY, seq, func, args, cb_id))
        return DauSimCallback(self, cb_id)

    def register_rwsynch_callback(self, func: Callable, *args: Any) -> DauSimCallback:
        cb_id = self._alloc_cb_id()
        seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._callback_queue, (self._sim_time, _PHASE_READWRITE, seq, func, args, cb_id))
        return DauSimCallback(self, cb_id)

    def register_nextstep_callback(self, func: Callable, *args: Any) -> DauSimCallback:
        cb_id = self._alloc_cb_id()
        seq = self._seq_counter
        self._seq_counter += 1
        # Schedule at the start of the next time step
        heapq.heappush(self._callback_queue, (self._sim_time + 1, _PHASE_NORMAL, seq, func, args, cb_id))
        return DauSimCallback(self, cb_id)

    def register_value_change_callback(self, handle: DauSimHandle, func: Callable, edge: int, *args: Any) -> DauSimCallback:
        cb_id = self._alloc_cb_id()
        sig_name = handle._name
        if handle._parent_signal is not None:
            sig_name = handle._parent_signal
        self._vch_callbacks.setdefault(sig_name, []).append((edge, func, args, cb_id))
        return DauSimCallback(self, cb_id)

    # Build handle hierarchy
    def build_handle_hierarchy(self) -> DauSimHandle:
        """Create the root DauSimHandle with children for all ports and signals."""
        children: dict[str, DauSimHandle] = {}

        # Ports first
        for p in self._module.ports:
            children[p.name] = DauSimHandle(self, p.name, f"{self._module.name}.{p.name}", shape=p.shape)

        # Internal signals
        for s in self._module.signals:
            if s.name not in children:
                children[s.name] = DauSimHandle(self, s.name, f"{self._module.name}.{s.name}", shape=s.shape)

        self._root_handle = DauSimHandle(
            self,
            self._module.name,
            self._module.name,
            is_scope=True,
            children=children,
        )
        return self._root_handle

    # Design evaluation helpers
    def _settle(self, comb_only: bool = False) -> list[str]:
        """Detect clock edges, execute sequential logic, settle combinational.

        Compares current ``_signals`` against ``_prev_signals`` to detect
        clock edges.  Does **not** auto-toggle clocks — cocotb drives them.

        When *comb_only* is ``True`` (used during delta cycles), skip
        sequential block evaluation — only settle combinational logic.
        Sequential blocks should fire at most once per simulation time step,
        matching real HDL non-blocking assignment (NBA) semantics.

        Returns:
            List of domain names whose seq blocks fired.
        """
        fired_domains: list[str] = []

        if self._has_seq and not comb_only:
            # Detect clock edges from signal changes
            for dname, dinfo in self._domain_info.items():
                clk_sig = dinfo["clk_signal"]
                old_clk = self._prev_signals.get(clk_sig, 0)
                new_clk = self._signals.get(clk_sig, 0)
                if old_clk != new_clk and _edge_fires(dinfo["edge"], old_clk, new_clk):
                    fired_domains.append(dname)

            # Async reset
            for dname, dinfo in self._domain_info.items():
                rst_sig = dinfo["rst_signal"]
                if rst_sig is None or dinfo["rst_style"] != ResetStyle.ASYNC:
                    continue
                rst_val = self._signals.get(rst_sig, 0)
                rst_active = rst_val if dinfo["rst_active_high"] else (not rst_val)
                if rst_active:
                    for sig_name in dinfo["written_signals"]:
                        self._signals[sig_name] = mask_value(self._init_values.get(sig_name, 0), self._shapes[sig_name])
                    if dname in fired_domains:
                        fired_domains.remove(dname)

            # Seq blocks on active edges
            for dname in fired_domains:
                dinfo = self._domain_info[dname]
                rst_sig = dinfo["rst_signal"]
                if rst_sig is not None and dinfo["rst_style"] == ResetStyle.SYNC:
                    rst_val = self._signals.get(rst_sig, 0)
                    rst_active = rst_val if dinfo["rst_active_high"] else (not rst_val)
                    if rst_active:
                        for sig_name in dinfo["written_signals"]:
                            self._signals[sig_name] = mask_value(self._init_values.get(sig_name, 0), self._shapes[sig_name])
                        continue
                for sb in dinfo["seq_blocks"]:
                    _exec_stmts(sb.stmts, self._signals, self._shapes)

            # Memory writes
            if self._module.memories and fired_domains:
                _exec_mem_writes(
                    self._mem_state,
                    self._module.memories,
                    self._signals,
                    fired_domains,
                )

            # Memory reads
            if self._module.memories:
                _exec_mem_reads(
                    self._mem_state,
                    self._module.memories,
                    self._signals,
                    fired_domains,
                )

        # Settle combinational logic
        for assignment in self._comb_order:
            _exec_stmts(assignment.stmts, self._signals, self._shapes)

        return fired_domains

    def _fire_phase(self, phase: int) -> None:
        """Fire all callbacks at ``self._sim_time`` with the given *phase*."""
        while self._callback_queue and self._callback_queue[0][0] == self._sim_time and self._callback_queue[0][1] == phase:
            _, _, _, func, args, cb_id = heapq.heappop(self._callback_queue)
            if cb_id in self._active_callbacks:
                self._active_callbacks.discard(cb_id)
                func(*args)

    def _has_phase_at_current_time(self, phase: int) -> bool:
        """Check whether the queue has callbacks at ``_sim_time`` with *phase*."""
        return bool(self._callback_queue) and self._callback_queue[0][0] == self._sim_time and self._callback_queue[0][1] == phase

    def _check_value_changes(self) -> None:
        """Detect signal value changes and fire VCH callbacks."""
        callbacks_to_fire: list[tuple[Callable, tuple]] = []

        for sig_name, watchers in list(self._vch_callbacks.items()):
            old_val = self._prev_signals.get(sig_name, 0)
            new_val = self._signals.get(sig_name, 0)
            if old_val == new_val:
                continue

            remaining: list[tuple[int, Callable, tuple, int]] = []
            for edge_type, func, args, cb_id in watchers:
                if cb_id not in self._active_callbacks:
                    continue
                fire = False
                if edge_type == VALUE_CHANGE:
                    fire = True
                elif edge_type == RISING:
                    fire = old_val == 0 and new_val != 0
                elif edge_type == FALLING:
                    fire = old_val != 0 and new_val == 0
                if fire:
                    callbacks_to_fire.append((func, args))
                    self._active_callbacks.discard(cb_id)
                    # VCH callbacks are one-shot; don't keep
                else:
                    remaining.append((edge_type, func, args, cb_id))

            if remaining:
                self._vch_callbacks[sig_name] = remaining
            else:
                del self._vch_callbacks[sig_name]

        # Fire VCH callbacks
        for func, args in callbacks_to_fire:
            func(*args)

    def step(self) -> bool:
        """Execute one simulation time step.

        Implements Verilog-style non-blocking assignment (NBA) semantics:
        sequential block outputs are staged and applied AFTER value-change
        callbacks fire, so cocotb test coroutines see the pre-NBA values
        (matching real HDL simulator behavior where VCH fires in the
        Active/Observed region before NBA updates take effect).

        Returns ``True`` if there may be more work, ``False`` if the
        callback queue is empty and there are no VCH watchers.
        """
        if not self._callback_queue and not self._vch_callbacks:
            return False

        # 1. Advance sim_time to earliest callback ---
        if self._callback_queue:
            next_time = self._callback_queue[0][0]
        else:
            # Only VCH callbacks remain — advance by one tick so clocks
            # can be toggled externally.
            next_time = self._sim_time + 1
        self._sim_time = next_time

        # 2. Fire Normal (timed) callbacks ---
        self._fire_phase(_PHASE_NORMAL)

        # 3. Fire ReadWrite callbacks (applies scheduled writes) ---
        self._fire_phase(_PHASE_READWRITE)

        # 4. Settle design (detect clock edges, run seq, settle comb) ---
        #    Identify which signals are written by sequential blocks so we
        #    can stage their updates (NBA semantics).
        pre_settle = dict(self._signals)
        self._settle()

        # Identify sequential-block signal changes (NBA candidates)
        seq_written: set[str] = set()
        if self._has_seq:
            for dinfo in self._domain_info.values():
                seq_written.update(dinfo.get("written_signals", ()))

        nba: dict[str, int] = {}
        for sig in seq_written:
            new_val = self._signals.get(sig, 0)
            old_val = pre_settle.get(sig, 0)
            if new_val != old_val:
                nba[sig] = new_val
                # Restore pre-settle value so VCH sees pre-NBA state
                self._signals[sig] = old_val

        # 5. Fire value-change callbacks (test sees pre-NBA values) ---
        self._check_value_changes()

        # 5b. Snapshot prev_signals for post-NBA VCH detection
        self._prev_signals = dict(self._signals)

        # 5c. Apply NBA (sequential block outputs take effect) ---
        if nba:
            self._signals.update(nba)
            # Re-settle combinational logic with post-NBA values
            for assignment in self._comb_order:
                _exec_stmts(assignment.stmts, self._signals, self._shapes)
            # Fire VCH for NBA-induced changes
            self._check_value_changes()
            self._prev_signals = dict(self._signals)

        # 6. Delta cycles: if VCH handlers registered new ReadWrite
        #    callbacks at the current time, process them.
        #    Sequential blocks are NOT re-evaluated (comb_only=True)
        #    matching real HDL NBA semantics.
        delta_limit = 100
        for _ in range(delta_limit):
            if not self._has_phase_at_current_time(_PHASE_READWRITE):
                break
            self._fire_phase(_PHASE_READWRITE)
            self._settle(comb_only=True)
            self._check_value_changes()
            self._prev_signals = dict(self._signals)

        # 7. Fire ReadOnly callbacks ---
        self._fire_phase(_PHASE_READONLY)

        # 8. Snapshot prev_signals for next step ---
        self._prev_signals = dict(self._signals)

        return True

    def run(self, max_steps: int = 10_000_000) -> None:
        """Run the simulation loop until no more callbacks or *max_steps*."""
        self._running = True
        for _ in range(max_steps):
            if not self._running:
                break
            if not self.step():
                break
        self._running = False

    def stop(self) -> None:
        self._running = False


# Simulator module — drop-in replacement for cocotb.simulator


def _create_simulator_module(engine: SimulationEngine) -> types.ModuleType:
    """Create a module object that mimics ``cocotb.simulator``."""
    mod = types.ModuleType("cocotb.simulator")
    mod.__package__ = "cocotb"

    # Constants
    mod.MODULE = MODULE
    mod.LOGIC = LOGIC
    mod.LOGIC_ARRAY = LOGIC_ARRAY
    mod.INTEGER = INTEGER
    mod.REAL = REAL
    mod.ENUM = ENUM
    mod.STRING = STRING
    mod.GENARRAY = GENARRAY
    mod.STRUCTURE = STRUCTURE
    mod.NETARRAY = NETARRAY
    mod.MEMORY = MEMORY
    mod.PACKED_STRUCTURE = PACKED_STRUCTURE
    mod.UNKNOWN = UNKNOWN
    mod.PACKAGE = PACKAGE
    mod.OBJECTS = OBJECTS
    mod.DRIVERS = DRIVERS
    mod.LOADS = LOADS
    mod.RISING = RISING
    mod.FALLING = FALLING
    mod.VALUE_CHANGE = VALUE_CHANGE
    mod.RANGE_UP = RANGE_UP
    mod.RANGE_DOWN = RANGE_DOWN
    mod.RANGE_NO_DIR = RANGE_NO_DIR

    # Types
    mod.gpi_sim_hdl = DauSimHandle
    mod.gpi_cb_hdl = DauSimCallback
    mod.gpi_iterator_hdl = DauSimIterator

    # Functions
    def register_timed_callback(time: int, func: Callable, *args: Any) -> DauSimCallback:
        return engine.register_timed_callback(time, func, *args)

    def register_readonly_callback(func: Callable, *args: Any) -> DauSimCallback:
        return engine.register_readonly_callback(func, *args)

    def register_rwsynch_callback(func: Callable, *args: Any) -> DauSimCallback:
        return engine.register_rwsynch_callback(func, *args)

    def register_nextstep_callback(func: Callable, *args: Any) -> DauSimCallback:
        return engine.register_nextstep_callback(func, *args)

    def register_value_change_callback(signal: DauSimHandle, func: Callable, edge: int, *args: Any) -> DauSimCallback:
        return engine.register_value_change_callback(signal, func, edge, *args)

    def get_root_handle(name: str | None = None) -> DauSimHandle:  # noqa: ARG001 (cocotb simulator API)
        return engine._root_handle

    def get_sim_time() -> tuple[int, int]:
        t = engine._sim_time
        return (t & 0xFFFFFFFF, (t >> 32) & 0xFFFFFFFF)

    def get_precision() -> int:
        return engine._time_precision

    def get_simulator_product() -> str:
        return "dau-sim"

    def get_simulator_version() -> str:
        return "0.1.0"

    def is_running() -> bool:
        return engine._running

    def stop_simulator() -> None:
        engine.stop()

    def clock_create(signal: DauSimHandle) -> None:
        pass  # Not needed — cocotb.clock.Clock handles this via Timer triggers

    def initialize_logger(log_level: int) -> None:
        pass  # Logging handled by Python

    def set_gpi_log_level(level: int) -> None:
        pass

    def package_iterate() -> None:
        return None

    def set_sim_event_callback(func: Callable) -> None:
        pass  # We handle sim events via Python exceptions

    mod.register_timed_callback = register_timed_callback
    mod.register_readonly_callback = register_readonly_callback
    mod.register_rwsynch_callback = register_rwsynch_callback
    mod.register_nextstep_callback = register_nextstep_callback
    mod.register_value_change_callback = register_value_change_callback
    mod.get_root_handle = get_root_handle
    mod.get_sim_time = get_sim_time
    mod.get_precision = get_precision
    mod.get_simulator_product = get_simulator_product
    mod.get_simulator_version = get_simulator_version
    mod.is_running = is_running
    mod.stop_simulator = stop_simulator
    mod.clock_create = clock_create
    mod.initialize_logger = initialize_logger
    mod.set_gpi_log_level = set_gpi_log_level
    mod.package_iterate = package_iterate
    mod.set_sim_event_callback = set_sim_event_callback

    # GpiClock placeholder
    class GpiClock:
        pass

    mod.GpiClock = GpiClock

    return mod


def run_cocotb(
    design: Any,
    test_module: str,
    *,
    time_precision: int = -12,
) -> None:
    """Run cocotb tests against a dau-sim simulated design.

    Args:
        design: An Amaranth ``Elaboratable`` or a dau-sim IR ``Module``.
        test_module: Python module name containing ``@cocotb.test()`` functions.
        time_precision: Simulator time precision as power of 10 (default -12 = 1ps).
    """
    # Step 1: Lower design to IR Module ---
    if isinstance(design, Module):
        ir_module = design
    else:
        # Try Amaranth
        try:
            from dau_sim.frontends.amaranth_frontend import lower_amaranth_design

            ir_module = lower_amaranth_design(design)
        except (ImportError, TypeError):
            raise TypeError(f"Cannot lower design of type {type(design).__name__}. Pass an Amaranth Elaboratable or a dau-sim IR Module.")

    # Step 2: Create simulation engine ---
    engine = SimulationEngine(ir_module, time_precision=time_precision)

    # Build handle hierarchy
    engine.build_handle_hierarchy()

    # Step 3: Patch cocotb.simulator and run ---
    sim_module = _create_simulator_module(engine)
    _run_with_patched_simulator(engine, sim_module, test_module)


def _run_with_patched_simulator(
    engine: SimulationEngine,
    sim_module: types.ModuleType,
    test_module: str,
) -> None:
    """Patch sys.modules and run cocotb's regression manager."""
    # isort: off
    import cocotb
    import cocotb.handle  # must precede cocotb._gpi_triggers (circular import)
    import cocotb._gpi_triggers
    import cocotb.simtime
    # isort: on

    # Save originals
    orig_simulator = sys.modules.get("cocotb.simulator")
    orig_cocotb_sim = getattr(cocotb, "simulator", None)
    orig_triggers_sim = getattr(cocotb._gpi_triggers, "simulator", None)
    orig_handle_sim = getattr(cocotb.handle, "simulator", None)
    orig_simtime_sim = getattr(cocotb.simtime, "simulator", None)
    orig_is_simulation = getattr(cocotb, "is_simulation", False)
    orig_top = getattr(cocotb, "top", None)
    orig_scheduler = getattr(cocotb, "_scheduler_inst", None)
    orig_regression = getattr(cocotb, "_regression_manager", None)
    orig_env_modules = os.environ.get("COCOTB_TEST_MODULES")

    try:
        # Patch cocotb.simulator into sys.modules and all modules that
        # bind it as a local name via ``from cocotb import simulator``.
        sys.modules["cocotb.simulator"] = sim_module
        cocotb.simulator = sim_module
        cocotb._gpi_triggers.simulator = sim_module
        cocotb.handle.simulator = sim_module
        cocotb.simtime.simulator = sim_module

        # Set cocotb global state
        cocotb.is_simulation = True
        cocotb.argv = []
        cocotb.plusargs = {}
        cocotb.SIM_NAME = "dau-sim"
        cocotb.SIM_VERSION = "0.1.0"
        cocotb.RANDOM_SEED = int(time.time())
        random.seed(cocotb.RANDOM_SEED)
        cocotb.packages = types.SimpleNamespace()

        # Setup logging (if not already done)
        if not hasattr(cocotb, "log") or cocotb.log is None:
            cocotb.log = logging.getLogger("test")
            cocotb.log.setLevel(logging.INFO)

        # Setup root handle
        cocotb.top = cocotb.handle._make_sim_object(engine._root_handle)

        # Init simtime precision
        cocotb.simtime._init()

        # Create scheduler
        from cocotb._scheduler import Scheduler

        cocotb._scheduler_inst = Scheduler()

        # Discover and run tests
        from cocotb.regression import RegressionManager

        cocotb._regression_manager = RegressionManager()

        os.environ["COCOTB_TEST_MODULES"] = test_module
        cocotb._regression_manager.setup_pytest_assertion_rewriting()
        cocotb._regression_manager.discover_tests(test_module)

        # start_regression() runs the first test synchronously until it
        # suspends (e.g. ``await Timer(...)``).  After it returns, all GPI
        # callbacks that the test registered are sitting in our engine's
        # callback queue.
        cocotb._regression_manager.start_regression()

        # Drive the simulation loop — each step fires callbacks that
        # re-enter cocotb's scheduler via ``_sim_react``.
        engine._running = True
        engine.run()

    finally:
        # Clear the handle cache so it doesn't leak across runs
        cocotb.handle._handle2obj.clear()

        # Restore simulator references in all patched modules
        if orig_simulator is not None:
            sys.modules["cocotb.simulator"] = orig_simulator
            cocotb.simulator = orig_cocotb_sim
        else:
            sys.modules.pop("cocotb.simulator", None)
            if hasattr(cocotb, "simulator"):
                delattr(cocotb, "simulator")
        cocotb._gpi_triggers.simulator = orig_triggers_sim
        cocotb.handle.simulator = orig_handle_sim
        cocotb.simtime.simulator = orig_simtime_sim
        cocotb.is_simulation = orig_is_simulation
        cocotb.top = orig_top
        cocotb._scheduler_inst = orig_scheduler
        cocotb._regression_manager = orig_regression
        if orig_env_modules is not None:
            os.environ["COCOTB_TEST_MODULES"] = orig_env_modules
        else:
            os.environ.pop("COCOTB_TEST_MODULES", None)
