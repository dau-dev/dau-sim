from __future__ import annotations

import math
from datetime import datetime, timedelta

import csp
from csp import ts

from dau_sim.compiler.depanalysis import (
    Assignment,
    build_assignments,
    collect_stmt_writes,
    topological_sort,
)
from dau_sim.compiler.eval import eval_expr, mask_value
from dau_sim.compiler.eval4 import eval_expr_4
from dau_sim.ir.module import Module, SeqBlock
from dau_sim.ir.stmt import Assert, Assign, Delay, Finish, IfElse, Print, Stmt, Switch
from dau_sim.ir.types import EdgePolarity, FourState, NetKind, ResetStyle, Shape


class SimulationFinish(Exception):
    """Raised when a $finish statement is executed."""

    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        super().__init__(f"$finish({exit_code})")


def _exec_stmts(
    stmts: tuple[Stmt, ...],
    signals: dict[str, int],
    shapes: dict[str, Shape],
) -> None:
    """Execute a list of statements, mutating signals in-place."""
    for stmt in stmts:
        _exec_stmt(stmt, signals, shapes)


def _exec_stmt(
    stmt: Stmt,
    current: dict[str, int],
    shapes: dict[str, Shape],
) -> None:
    """Execute a single statement, mutating current signal values."""
    if isinstance(stmt, Assign):
        val = eval_expr(stmt.value, current)
        val = mask_value(val, shapes[stmt.target])
        current[stmt.target] = val

    elif isinstance(stmt, IfElse):
        cond = eval_expr(stmt.cond, current)
        body = stmt.then_body if cond else stmt.else_body
        for s in body:
            _exec_stmt(s, current, shapes)

    elif isinstance(stmt, Switch):
        test = eval_expr(stmt.test, current)
        default_stmts: tuple[Stmt, ...] = ()
        matched = False
        for pattern, stmts in stmt.cases:
            if pattern is None:
                default_stmts = stmts
            elif pattern == test:
                matched = True
                for s in stmts:
                    _exec_stmt(s, current, shapes)
                break
        if not matched and default_stmts:
            for s in default_stmts:
                _exec_stmt(s, current, shapes)

    elif isinstance(stmt, Assert):
        cond = eval_expr(stmt.cond, current)
        if not cond:
            msg = stmt.message or "assertion failed"
            raise AssertionError(msg)

    elif isinstance(stmt, Print):
        args = [eval_expr(a, current) for a in stmt.args]
        print(stmt.format_str.format(*args))

    elif isinstance(stmt, Finish):
        raise SimulationFinish(stmt.exit_code)

    elif isinstance(stmt, Delay):
        pass  # Delay is a no-op in synchronous execution; handled by InitBlock runner


def _exec_stmts_4(
    stmts: tuple[Stmt, ...],
    signals: dict[str, FourState],
    shapes: dict[str, Shape],
) -> None:
    """Execute statements under four-state semantics."""
    for stmt in stmts:
        _exec_stmt_4(stmt, signals, shapes)


def _exec_stmt_4(
    stmt: Stmt,
    current: dict[str, FourState],
    shapes: dict[str, Shape],
) -> None:
    """Execute a single statement under four-state semantics."""
    if isinstance(stmt, Assign):
        val = eval_expr_4(stmt.value, current)
        current[stmt.target] = FourState(shape=shapes[stmt.target], aval=val.aval, bval=val.bval)

    elif isinstance(stmt, IfElse):
        cond = eval_expr_4(stmt.cond, current)
        if cond.has_unknown:
            # X condition: execute both branches (conservative)
            for s in stmt.then_body:
                _exec_stmt_4(s, current, shapes)
            for s in stmt.else_body:
                _exec_stmt_4(s, current, shapes)
        else:
            body = stmt.then_body if cond.aval else stmt.else_body
            for s in body:
                _exec_stmt_4(s, current, shapes)

    elif isinstance(stmt, Switch):
        cond = eval_expr_4(stmt.test, current)
        if cond.has_unknown:
            # X test: execute all cases (conservative)
            for _, stmts in stmt.cases:
                for s in stmts:
                    _exec_stmt_4(s, current, shapes)
        else:
            test = cond.aval
            default_stmts: tuple[Stmt, ...] = ()
            matched = False
            for pattern, stmts in stmt.cases:
                if pattern is None:
                    default_stmts = stmts
                elif pattern == test:
                    matched = True
                    for s in stmts:
                        _exec_stmt_4(s, current, shapes)
                    break
            if not matched and default_stmts:
                for s in default_stmts:
                    _exec_stmt_4(s, current, shapes)

    elif isinstance(stmt, Assert):
        cond = eval_expr_4(stmt.cond, current)
        if cond.is_fully_defined and cond.aval == 0:
            msg = stmt.message or "assertion failed"
            raise AssertionError(msg)

    elif isinstance(stmt, Print):
        args = []
        for a in stmt.args:
            v = eval_expr_4(a, current)
            args.append(v.to_int if v.is_fully_defined else "x")
        print(stmt.format_str.format(*args))

    elif isinstance(stmt, Finish):
        raise SimulationFinish(stmt.exit_code)

    elif isinstance(stmt, Delay):
        pass  # No-op in synchronous execution


def _build_domain_info(
    module: Module,
) -> dict[str, dict]:
    """Precompute per-domain metadata for the sequential engine.

    Returns a dict keyed by domain name with:
        clk_signal: str           - signal name of the clock
        edge: EdgePolarity        - which edge triggers sequential blocks
        rst_signal: str | None    - signal name of reset (or None)
        rst_style: ResetStyle     - sync or async
        rst_active_high: bool
        seq_blocks: list[SeqBlock]
        written_signals: set[str] - signals written by this domain's seq blocks
    """
    # Group seq blocks by domain
    domain_seq: dict[str, list[SeqBlock]] = {}
    for sb in module.seq_blocks:
        domain_seq.setdefault(sb.domain, []).append(sb)

    info: dict[str, dict] = {}
    for cd in module.clock_domains:
        sbs = domain_seq.get(cd.name, [])
        written: set[str] = set()
        for sb in sbs:
            for stmt in sb.stmts:
                written |= collect_stmt_writes(stmt)
        info[cd.name] = {
            "clk_signal": cd.clk,
            "edge": cd.edge,
            "rst_signal": cd.rst,
            "rst_style": cd.rst_style,
            "rst_active_high": cd.rst_active_high,
            "seq_blocks": sbs,
            "written_signals": written,
        }
    return info


def _collect_init_values(module: Module) -> dict[str, int]:
    """Collect init values for all signals by name."""
    inits: dict[str, int] = {}
    for p in module.ports:
        inits[p.name] = p.signal.init
    for s in module.signals:
        inits[s.name] = s.init
    return inits


def _timedelta_ns(td: timedelta) -> int:
    """Convert timedelta to integer nanoseconds."""
    return max(1, round(td.total_seconds() * 1_000_000_000))


def _compute_half_period_ticks(
    clock_period: timedelta,
    clocks: dict[str, timedelta] | None,
    domain_info: dict[str, dict],
) -> int:
    """Compute per-domain half_period_ticks and return the GCD tick count.

    Each domain's half-period is expressed as an integer multiple of the
    finest simulation tick.  Returns the number of ticks that correspond
    to one GCD unit (always 1 — the GCD *is* the tick).
    """
    if clocks:
        periods_ns = {d: _timedelta_ns(clocks[d]) if d in clocks else _timedelta_ns(clock_period) for d in domain_info}
    else:
        cp_ns = _timedelta_ns(clock_period)
        periods_ns = {d: cp_ns for d in domain_info}

    half_ns = {d: max(1, p // 2) for d, p in periods_ns.items()}

    gcd_ns = 0
    for h in half_ns.values():
        gcd_ns = math.gcd(gcd_ns, h) if gcd_ns else h
    if gcd_ns == 0:
        gcd_ns = 1

    for dname, dinfo in domain_info.items():
        dinfo["half_period_ticks"] = half_ns[dname] // gcd_ns

    return gcd_ns


def _exec_mem_writes(
    mem_state: dict[str, list[int]],
    memories: tuple,
    signals: dict[str, int],
    shapes: dict[str, Shape],
    fired_domains: list[str],
) -> None:
    """Execute memory write ports for fired clock domains."""
    for mem in memories:
        for wp in mem.write_ports:
            if wp.domain not in fired_domains:
                continue
            en_val = signals.get(wp.en, 0)
            if not en_val:
                continue
            addr = signals.get(wp.addr, 0)
            if addr < 0 or addr >= mem.depth:
                continue
            data = signals.get(wp.data, 0)
            if wp.granularity > 0:
                bits_per_gran = wp.granularity
                n_grans = mem.shape.width // bits_per_gran
                for g in range(n_grans):
                    if (en_val >> g) & 1:
                        mask = ((1 << bits_per_gran) - 1) << (g * bits_per_gran)
                        mem_state[mem.name][addr] = (mem_state[mem.name][addr] & ~mask) | (data & mask)
            else:
                mem_state[mem.name][addr] = mask_value(data, mem.shape)


def _exec_mem_reads(
    mem_state: dict[str, list[int]],
    memories: tuple,
    signals: dict[str, int],
    fired_domains: list[str] | None,
) -> None:
    """Execute memory read ports.

    For combinational reads (domain is None), always execute.
    For synchronous reads, only execute on fired domains.
    Transparent reads forward write data from same-cycle writes.
    """
    for mem in memories:
        for rp in mem.read_ports:
            if rp.domain is None:
                # Combinational read: always active
                if rp.en and not signals.get(rp.en, 1):
                    continue
                addr = signals.get(rp.addr, 0)
                if 0 <= addr < mem.depth:
                    signals[rp.data] = mem_state[mem.name][addr]
            else:
                # Synchronous read: only on clock edge
                if fired_domains is not None and rp.domain in fired_domains:
                    if rp.en and not signals.get(rp.en, 1):
                        continue
                    addr = signals.get(rp.addr, 0)
                    if 0 <= addr < mem.depth:
                        val = mem_state[mem.name][addr]
                        # Check for transparent forwarding
                        for wp_idx in rp.transparent_for:
                            wp = mem.write_ports[wp_idx]
                            wp_en = signals.get(wp.en, 0)
                            wp_addr = signals.get(wp.addr, 0)
                            if wp_en and wp_addr == addr:
                                val = signals.get(wp.data, 0)
                                break
                        signals[rp.data] = val


def _edge_fires(edge: EdgePolarity, old_clk: int, new_clk: int) -> bool:
    """Check if a clock transition triggers the given edge polarity."""
    if edge == EdgePolarity.POSEDGE:
        return old_clk == 0 and new_clk == 1
    elif edge == EdgePolarity.NEGEDGE:
        return old_clk == 1 and new_clk == 0
    else:  # BOTH
        return old_clk != new_clk


@csp.node
def _sim_tick(
    tick: ts[bool],
    init_signals: ts[dict],
    comb_order: ts[object],
    seq_blocks: ts[object],
    shapes: ts[object],
    memories: ts[object],
    mem_init: ts[object],
) -> ts[dict]:
    """Combinational-only simulation node (no clock edge semantics).

    Used for modules with no sequential blocks.  Every timer tick
    evaluates all comb blocks and emits the signal dict.
    """
    with csp.state():
        s_signals: dict = {}
        s_comb_order: list = []
        s_seq_blocks: tuple = ()
        s_shapes: dict = {}
        s_initialized: bool = False
        s_finished: bool = False
        s_memories: tuple = ()
        s_mem_state: dict = {}

    if csp.ticked(init_signals):
        s_signals = dict(init_signals)
        s_initialized = True
    if csp.ticked(comb_order):
        s_comb_order = comb_order
    if csp.ticked(seq_blocks):
        s_seq_blocks = seq_blocks
    if csp.ticked(shapes):
        s_shapes = shapes
    if csp.ticked(memories):
        s_memories = memories
    if csp.ticked(mem_init):
        s_mem_state = {k: list(v) for k, v in mem_init.items()}

    if csp.ticked(tick) and s_initialized and not s_finished:
        try:
            # Sequential blocks (legacy path — no edge gating)
            for sb in s_seq_blocks:
                _exec_stmts(sb.stmts, s_signals, s_shapes)

            # Memory reads (combinational only, no fired domains)
            if s_memories:
                _exec_mem_reads(s_mem_state, s_memories, s_signals, [])

            # Combinational blocks in dependency order
            for assignment in s_comb_order:
                _exec_stmts(assignment.stmts, s_signals, s_shapes)
        except SimulationFinish:
            s_finished = True

        return dict(s_signals)


@csp.node
def _sim_tick_4(
    tick: ts[bool],
    init_signals: ts[dict],
    comb_order: ts[object],
    seq_blocks: ts[object],
    shapes: ts[object],
    memories: ts[object],
    mem_init: ts[object],
) -> ts[dict]:
    """Four-state combinational-only simulation node."""
    with csp.state():
        s_signals: dict = {}
        s_comb_order: list = []
        s_seq_blocks: tuple = ()
        s_shapes: dict = {}
        s_initialized: bool = False
        s_finished: bool = False
        s_memories: tuple = ()
        s_mem_state: dict = {}

    if csp.ticked(init_signals):
        s_signals = dict(init_signals)
        s_initialized = True
    if csp.ticked(comb_order):
        s_comb_order = comb_order
    if csp.ticked(seq_blocks):
        s_seq_blocks = seq_blocks
    if csp.ticked(shapes):
        s_shapes = shapes
    if csp.ticked(memories):
        s_memories = memories
    if csp.ticked(mem_init):
        s_mem_state = {k: list(v) for k, v in mem_init.items()}

    if csp.ticked(tick) and s_initialized and not s_finished:
        try:
            for sb in s_seq_blocks:
                _exec_stmts_4(sb.stmts, s_signals, s_shapes)

            # Memory reads (combinational only, no fired domains)
            if s_memories:
                _exec_mem_reads(s_mem_state, s_memories, s_signals, [])

            for assignment in s_comb_order:
                _exec_stmts_4(assignment.stmts, s_signals, s_shapes)
        except SimulationFinish:
            s_finished = True
        return dict(s_signals)


@csp.node
def _sim_engine_seq(
    tick: ts[bool],
    init_signals: ts[dict],
    comb_order: ts[object],
    domain_info: ts[object],
    shapes: ts[object],
    init_values: ts[object],
    memories: ts[object],
    mem_init: ts[object],
) -> ts[dict]:
    """Clock-aware simulation engine (two-state).

    Tracks per-domain clock phase, detects active edges, handles
    sync/async reset, and only fires seq blocks on the correct edge.
    Emits output only when at least one domain had an active edge.
    """
    with csp.state():
        s_signals: dict = {}
        s_comb_order: list = []
        s_domain_info: dict = {}
        s_shapes: dict = {}
        s_init_values: dict = {}
        s_initialized: bool = False
        s_finished: bool = False
        s_tick_count: int = 0
        s_clock_states: dict = {}  # domain_name -> current clk value (0 or 1)
        s_half_period_ticks: dict = {}  # domain_name -> int
        s_memories: tuple = ()
        s_mem_state: dict = {}

    if csp.ticked(init_signals):
        s_signals = dict(init_signals)
        s_initialized = True
    if csp.ticked(comb_order):
        s_comb_order = comb_order
    if csp.ticked(domain_info):
        s_domain_info = domain_info
        for dname, dinfo in s_domain_info.items():
            s_clock_states[dname] = 0
            s_half_period_ticks[dname] = dinfo.get("half_period_ticks", 1)
    if csp.ticked(shapes):
        s_shapes = shapes
    if csp.ticked(init_values):
        s_init_values = init_values
    if csp.ticked(memories):
        s_memories = memories
    if csp.ticked(mem_init):
        s_mem_state = {k: list(v) for k, v in mem_init.items()}

    if csp.ticked(tick) and s_initialized and not s_finished:
        s_tick_count += 1

        # Toggle clocks and detect edges
        fired_domains: list[str] = []
        for dname, dinfo in s_domain_info.items():
            hpt = s_half_period_ticks[dname]
            if s_tick_count % hpt == 0:
                old_clk = s_clock_states[dname]
                new_clk = 1 - old_clk
                s_clock_states[dname] = new_clk
                # Update clock signal in state
                s_signals[dinfo["clk_signal"]] = new_clk

                if _edge_fires(dinfo["edge"], old_clk, new_clk):
                    fired_domains.append(dname)

        # Async reset (fires regardless of clock edge)
        for dname, dinfo in s_domain_info.items():
            rst_sig = dinfo["rst_signal"]
            if rst_sig is None or dinfo["rst_style"] != ResetStyle.ASYNC:
                continue
            rst_val = s_signals.get(rst_sig, 0)
            rst_active = rst_val if dinfo["rst_active_high"] else (not rst_val)
            if rst_active:
                for sig_name in dinfo["written_signals"]:
                    s_signals[sig_name] = mask_value(s_init_values.get(sig_name, 0), s_shapes[sig_name])
                # Remove from fired list — reset overrides normal execution
                if dname in fired_domains:
                    fired_domains.remove(dname)

        # Sequential blocks on active edges
        try:
            for dname in fired_domains:
                dinfo = s_domain_info[dname]

                # Sync reset check
                rst_sig = dinfo["rst_signal"]
                if rst_sig is not None and dinfo["rst_style"] == ResetStyle.SYNC:
                    rst_val = s_signals.get(rst_sig, 0)
                    rst_active = rst_val if dinfo["rst_active_high"] else (not rst_val)
                    if rst_active:
                        for sig_name in dinfo["written_signals"]:
                            s_signals[sig_name] = mask_value(s_init_values.get(sig_name, 0), s_shapes[sig_name])
                        continue

                # Normal execution
                for sb in dinfo["seq_blocks"]:
                    _exec_stmts(sb.stmts, s_signals, s_shapes)

            # Memory writes (after seq blocks, before comb settle)
            if s_memories and fired_domains:
                _exec_mem_writes(s_mem_state, s_memories, s_signals, s_shapes, fired_domains)

            # Memory reads (synchronous reads on fired domains, combinational reads always)
            if s_memories:
                _exec_mem_reads(s_mem_state, s_memories, s_signals, fired_domains)

            # Settle combinational logic
            if fired_domains:
                for assignment in s_comb_order:
                    _exec_stmts(assignment.stmts, s_signals, s_shapes)
        except SimulationFinish:
            s_finished = True

        if fired_domains:
            return dict(s_signals)


@csp.node
def _sim_engine_seq_4(
    tick: ts[bool],
    init_signals: ts[dict],
    comb_order: ts[object],
    domain_info: ts[object],
    shapes: ts[object],
    init_values: ts[object],
    memories: ts[object],
    mem_init: ts[object],
) -> ts[dict]:
    """Clock-aware simulation engine (four-state)."""
    with csp.state():
        s_signals: dict = {}
        s_comb_order: list = []
        s_domain_info: dict = {}
        s_shapes: dict = {}
        s_init_values: dict = {}
        s_initialized: bool = False
        s_finished: bool = False
        s_tick_count: int = 0
        s_clock_states: dict = {}
        s_half_period_ticks: dict = {}
        s_memories: tuple = ()
        s_mem_state: dict = {}

    if csp.ticked(init_signals):
        s_signals = dict(init_signals)
        s_initialized = True
    if csp.ticked(comb_order):
        s_comb_order = comb_order
    if csp.ticked(domain_info):
        s_domain_info = domain_info
        for dname, dinfo in s_domain_info.items():
            s_clock_states[dname] = 0
            s_half_period_ticks[dname] = dinfo.get("half_period_ticks", 1)
    if csp.ticked(shapes):
        s_shapes = shapes
    if csp.ticked(init_values):
        s_init_values = init_values
    if csp.ticked(memories):
        s_memories = memories
    if csp.ticked(mem_init):
        s_mem_state = {k: list(v) for k, v in mem_init.items()}

    if csp.ticked(tick) and s_initialized and not s_finished:
        s_tick_count += 1

        fired_domains: list[str] = []
        for dname, dinfo in s_domain_info.items():
            hpt = s_half_period_ticks[dname]
            if s_tick_count % hpt == 0:
                old_clk = s_clock_states[dname]
                new_clk = 1 - old_clk
                s_clock_states[dname] = new_clk
                s_signals[dinfo["clk_signal"]] = FourState.from_int(new_clk, Shape(1))
                if _edge_fires(dinfo["edge"], old_clk, new_clk):
                    fired_domains.append(dname)

        # Async reset
        for dname, dinfo in s_domain_info.items():
            rst_sig = dinfo["rst_signal"]
            if rst_sig is None or dinfo["rst_style"] != ResetStyle.ASYNC:
                continue
            rst_fs = s_signals.get(rst_sig, FourState.from_int(0, Shape(1)))
            if rst_fs.is_fully_defined:
                rst_active = rst_fs.aval if dinfo["rst_active_high"] else (not rst_fs.aval)
            else:
                rst_active = False
            if rst_active:
                for sig_name in dinfo["written_signals"]:
                    s_signals[sig_name] = FourState.from_int(s_init_values.get(sig_name, 0), s_shapes[sig_name])
                if dname in fired_domains:
                    fired_domains.remove(dname)

        # Seq blocks
        try:
            for dname in fired_domains:
                dinfo = s_domain_info[dname]
                rst_sig = dinfo["rst_signal"]
                if rst_sig is not None and dinfo["rst_style"] == ResetStyle.SYNC:
                    rst_fs = s_signals.get(rst_sig, FourState.from_int(0, Shape(1)))
                    if rst_fs.is_fully_defined:
                        rst_active = rst_fs.aval if dinfo["rst_active_high"] else (not rst_fs.aval)
                    else:
                        rst_active = False
                    if rst_active:
                        for sig_name in dinfo["written_signals"]:
                            s_signals[sig_name] = FourState.from_int(s_init_values.get(sig_name, 0), s_shapes[sig_name])
                        continue
                for sb in dinfo["seq_blocks"]:
                    _exec_stmts_4(sb.stmts, s_signals, s_shapes)

            # Memory writes (after seq blocks, before comb settle)
            if s_memories and fired_domains:
                _exec_mem_writes(s_mem_state, s_memories, s_signals, s_shapes, fired_domains)

            # Memory reads (synchronous reads on fired domains, combinational reads always)
            if s_memories:
                _exec_mem_reads(s_mem_state, s_memories, s_signals, fired_domains)

            # Settle comb
            if fired_domains:
                for assignment in s_comb_order:
                    _exec_stmts_4(assignment.stmts, s_signals, s_shapes)
        except SimulationFinish:
            s_finished = True

        if fired_domains:
            return dict(s_signals)


@csp.node
def _extract_signal(all_signals: ts[dict], name: ts[str]) -> ts[int]:
    """Extract a single signal's value from the aggregate dict."""
    with csp.state():
        s_name: str = ""

    if csp.ticked(name):
        s_name = name

    if csp.ticked(all_signals) and s_name:
        if s_name in all_signals:
            return all_signals[s_name]


@csp.node
def _extract_signal_4(all_signals: ts[dict], name: ts[str]) -> ts[object]:
    """Extract a single FourState signal value."""
    with csp.state():
        s_name: str = ""

    if csp.ticked(name):
        s_name = name

    if csp.ticked(all_signals) and s_name:
        if s_name in all_signals:
            return all_signals[s_name]


class CompiledModule:
    """A compiled CSP graph for a single IR module."""

    def __init__(
        self,
        module: Module,
        comb_order: list[Assignment],
        four_state: bool = False,
    ):
        self.module = module
        self._comb_order = comb_order
        self._four_state = four_state
        self._shapes: dict[str, Shape] = {}
        self._net_kinds: dict[str, NetKind] = {}
        for p in module.ports:
            self._shapes[p.name] = p.shape
            self._net_kinds[p.name] = p.signal.net_kind
        for s in module.signals:
            self._shapes[s.name] = s.shape
            self._net_kinds[s.name] = s.net_kind
        # Phase 2: precompute domain info and init values
        self._domain_info = _build_domain_info(module)
        self._init_values = _collect_init_values(module)
        self._has_seq = (
            len(module.seq_blocks) > 0
            or any(mem.write_ports for mem in module.memories)
            or any(rp.domain is not None for mem in module.memories for rp in mem.read_ports)
        )
        self._has_memories = len(module.memories) > 0

        # Build memory init state
        self._mem_init: dict[str, list[int]] = {}
        self._mem_defs = {mem.name: mem for mem in module.memories}
        for mem in module.memories:
            init_data = list(mem.init) if mem.init else []
            # Pad to depth with zeros
            init_data.extend([0] * (mem.depth - len(init_data)))
            self._mem_init[mem.name] = init_data
            # Register memory port signals in shapes if not already present
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

    def write_vcd(
        self,
        path: str,
        traces: dict[str, list[tuple[datetime, int]]],
        *,
        timescale: str = "1ns",
        signals: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Write simulation traces to a VCD file.

        Convenience wrapper around ``dau_sim.adapters.vcd.write_vcd``
        that automatically provides the module metadata.

        *signals* and *exclude* accept glob patterns for filtering.
        """
        from dau_sim.adapters.vcd import write_vcd as _write_vcd

        if signals is not None or exclude is not None:
            from dau_sim.adapters.selectors import select_signals

            traces = select_signals(traces, include=signals, exclude=exclude)
        _write_vcd(path, traces, module=self.module, timescale=timescale)

    def traces_to_vcd(
        self,
        traces: dict[str, list[tuple[datetime, int]]],
        *,
        timescale: str = "1ns",
        signals: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> str:
        """Convert simulation traces to a VCD string.

        *signals* and *exclude* accept glob patterns for filtering.
        """
        from dau_sim.adapters.vcd import traces_to_vcd as _traces_to_vcd

        if signals is not None or exclude is not None:
            from dau_sim.adapters.selectors import select_signals

            traces = select_signals(traces, include=signals, exclude=exclude)
        return _traces_to_vcd(traces, module=self.module, timescale=timescale)

    def run_testbench(
        self,
        fn,
        clock_period: timedelta = timedelta(microseconds=1),
        clocks: dict[str, timedelta] | None = None,
        max_cycles: int = 10000,
    ):
        """Run a testbench function against this compiled module.

        The function receives a :class:`~dau_sim.testbench.TestbenchContext`
        and can call ``ctx.set()``, ``ctx.get()``, ``ctx.tick()``, and
        assertion methods to drive and verify the design.

        Returns a :class:`~dau_sim.testbench.TestbenchResult`.
        """
        from dau_sim.testbench import TestbenchContext, TestbenchResult

        ctx = TestbenchContext(self, clock_period, clocks, max_cycles)
        result = TestbenchResult(ctx)
        try:
            fn(ctx)
            result.passed = True
            result.cycle = ctx.cycle
            result.signals = dict(ctx._signals)
            result.history = {k: list(v) for k, v in ctx._history.items()}
        except Exception as e:
            result.passed = False
            result.error = e
            result.cycle = ctx.cycle
            result.signals = dict(ctx._signals)
            result.history = {k: list(v) for k, v in ctx._history.items()}
            raise
        return result

    def run(
        self,
        cycles: int = 10,
        clock_period: timedelta = timedelta(microseconds=1),
        inputs: dict[str, int] | None = None,
        clocks: dict[str, timedelta] | None = None,
    ) -> dict[str, list[tuple[datetime, int]]]:
        """Run simulation for N cycles and return signal traces.

        For modules with sequential logic (clock domains + seq blocks),
        ``cycles`` means full clock cycles.  The simulation timer runs at
        half-period resolution so clock edges are modeled correctly.

        For purely combinational modules, ``cycles`` means evaluation ticks
        (backward compatible with Phase 0/1).

        ``clocks`` optionally maps domain names to their clock period.  If
        omitted, all domains use ``clock_period``.
        """
        module = self.module
        shapes = self._shapes
        four_state = self._four_state
        comb_order = self._comb_order
        domain_info = self._domain_info
        init_values = self._init_values

        # Build initial signal values
        if four_state:
            init: dict = {}
            for p in module.ports:
                init[p.name] = FourState.from_int(p.signal.init, p.shape)
            for s in module.signals:
                init[s.name] = FourState.from_int(s.init, s.shape)
            if inputs:
                for name, val in inputs.items():
                    init[name] = FourState.from_int(val, shapes[name])
            # Initialize memory port signals
            for mem in module.memories:
                for rp in mem.read_ports:
                    for sname in (rp.addr, rp.data, rp.en):
                        if sname and sname not in init:
                            init[sname] = FourState.from_int(0, shapes[sname])
                for wp in mem.write_ports:
                    for sname in (wp.addr, wp.data, wp.en):
                        if sname and sname not in init:
                            init[sname] = FourState.from_int(0, shapes[sname])
        else:
            init = {}
            for p in module.ports:
                init[p.name] = p.signal.init
            for s in module.signals:
                init[s.name] = s.init
            if inputs:
                for name, val in inputs.items():
                    init[name] = mask_value(val, shapes[name])
            # Initialize memory port signals
            for mem in module.memories:
                for rp in mem.read_ports:
                    for sname in (rp.addr, rp.data, rp.en):
                        if sname and sname not in init:
                            init[sname] = 0
                for wp in mem.write_ports:
                    for sname in (wp.addr, wp.data, wp.en):
                        if sname and sname not in init:
                            init[sname] = 0

        all_names = list(shapes.keys())

        for ib in module.init_blocks:
            try:
                _exec_stmts(ib.stmts, init, shapes)
            except SimulationFinish:
                # $finish in init block → return single-point traces
                starttime = datetime(2000, 1, 1)
                return {name: [(starttime, init.get(name, 0))] for name in all_names}

        if self._has_seq:
            return self._run_sequential(
                cycles,
                clock_period,
                clocks,
                init,
                domain_info,
                init_values,
                comb_order,
                shapes,
                four_state,
                all_names,
            )
        else:
            return self._run_combinational(
                cycles,
                clock_period,
                init,
                comb_order,
                shapes,
                four_state,
                all_names,
            )

    def _run_combinational(
        self,
        cycles: int,
        clock_period: timedelta,
        init: dict,
        comb_order: list[Assignment],
        shapes: dict[str, Shape],
        four_state: bool,
        all_names: list[str],
    ) -> dict[str, list[tuple[datetime, int]]]:
        """Phase 0/1 mode: one evaluation per timer tick, no edge semantics."""
        seq_blocks = self.module.seq_blocks

        @csp.graph
        def sim_graph():
            tick = csp.timer(clock_period, True)
            init_edge = csp.const(init)
            comb_edge = csp.const(comb_order)
            seq_edge = csp.const(seq_blocks)
            shapes_edge = csp.const(shapes)
            mem_edge = csp.const(self.module.memories)
            mem_init_edge = csp.const(self._mem_init)

            if four_state:
                all_signals = _sim_tick_4(tick, init_edge, comb_edge, seq_edge, shapes_edge, mem_edge, mem_init_edge)
                for name in all_names:
                    sig_out = _extract_signal_4(all_signals, csp.const(name))
                    csp.add_graph_output(name, sig_out)
            else:
                all_signals = _sim_tick(tick, init_edge, comb_edge, seq_edge, shapes_edge, mem_edge, mem_init_edge)
                for name in all_names:
                    sig_out = _extract_signal(all_signals, csp.const(name))
                    csp.add_graph_output(name, sig_out)

        starttime = datetime(2000, 1, 1)
        endtime = starttime + clock_period * cycles

        raw = csp.run(sim_graph, starttime=starttime, endtime=endtime)
        return self._collect_traces(raw, all_names, four_state)

    def _run_sequential(
        self,
        cycles: int,
        clock_period: timedelta,
        clocks: dict[str, timedelta] | None,
        init: dict,
        domain_info: dict[str, dict],
        init_values: dict[str, int],
        comb_order: list[Assignment],
        shapes: dict[str, Shape],
        four_state: bool,
        all_names: list[str],
    ) -> dict[str, list[tuple[datetime, int]]]:
        """Phase 2 mode: clock-edge-driven sequential simulation."""
        # Compute per-domain half-period in ticks (mutates domain_info dicts)
        gcd_ns = _compute_half_period_ticks(clock_period, clocks, domain_info)

        # Total ticks: 2 half-periods per cycle * cycles, scaled by primary
        # domain's half_period_ticks.
        primary_name = next(iter(domain_info))
        primary_hpt = domain_info[primary_name]["half_period_ticks"]
        total_ticks = 2 * cycles * primary_hpt

        # Timer period — just for CSP timestamps (must be >= 1µs for timedelta)
        tick_period = timedelta(microseconds=max(1, gcd_ns // 1000))

        @csp.graph
        def sim_graph():
            tick = csp.timer(tick_period, True)
            init_edge = csp.const(init)
            comb_edge = csp.const(comb_order)
            domain_edge = csp.const(domain_info)
            shapes_edge = csp.const(shapes)
            init_vals_edge = csp.const(init_values)
            mem_edge = csp.const(self.module.memories)
            mem_init_edge = csp.const(self._mem_init)

            if four_state:
                all_signals = _sim_engine_seq_4(tick, init_edge, comb_edge, domain_edge, shapes_edge, init_vals_edge, mem_edge, mem_init_edge)
                for name in all_names:
                    sig_out = _extract_signal_4(all_signals, csp.const(name))
                    csp.add_graph_output(name, sig_out)
            else:
                all_signals = _sim_engine_seq(tick, init_edge, comb_edge, domain_edge, shapes_edge, init_vals_edge, mem_edge, mem_init_edge)
                for name in all_names:
                    sig_out = _extract_signal(all_signals, csp.const(name))
                    csp.add_graph_output(name, sig_out)

        starttime = datetime(2000, 1, 1)
        endtime = starttime + tick_period * total_ticks

        raw = csp.run(sim_graph, starttime=starttime, endtime=endtime)
        return self._collect_traces(raw, all_names, four_state)

    @staticmethod
    def _collect_traces(
        raw: dict,
        all_names: list[str],
        four_state: bool,
    ) -> dict[str, list[tuple[datetime, int]]]:
        result: dict[str, list[tuple[datetime, int]]] = {}
        for name in all_names:
            if name in raw:
                if four_state:
                    result[name] = [(t, v.to_int) for t, v in raw[name]]
                else:
                    result[name] = [(t, v) for t, v in raw[name]]
            else:
                result[name] = []
        return result


def compile_module(module: Module, four_state: bool = False) -> CompiledModule:
    """Compile an IR module into a CSP-executable CompiledModule.

    Args:
        module: The IR module to compile.
        four_state: If True, use four-state (X/Z) simulation semantics.

    Raises:
        CombLoopError: If combinational assignments form a cycle.
    """
    # Flatten hierarchy if needed
    if module.instances or module.submodules:
        from dau_sim.compiler.flatten import flatten_module

        module = flatten_module(module)

    # Build dependency-sorted comb block order
    stmts_list = [(i, cb.stmts) for i, cb in enumerate(module.comb_blocks)]
    assignments = build_assignments(stmts_list)
    comb_order = topological_sort(assignments)

    return CompiledModule(module, comb_order, four_state)
