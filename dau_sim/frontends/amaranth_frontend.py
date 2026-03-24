from __future__ import annotations

from amaranth.hdl import Const as AConst, Elaboratable, Signal as ASignal
from amaranth.hdl._ast import (
    Assign as AAssign,
    ClockSignal,
    Concat as AConcat,
    Initial,
    Operator,
    Part,
    ResetSignal,
    Slice as ASlice,
    Switch as ASwitch,
    SwitchValue,
)
from amaranth.hdl._ir import Fragment

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
from dau_sim.ir.module import (
    ClockDomain,
    CombBlock,
    Module,
    Port,
    SeqBlock,
    Signal,
)
from dau_sim.ir.stmt import Assign, Stmt, Switch
from dau_sim.ir.types import EdgePolarity, PortDirection, ResetStyle, Shape

__all__ = ("from_amaranth",)
_BINOP_MAP: dict[str, BinaryOp] = {
    "+": BinaryOp.ADD,
    "-": BinaryOp.SUB,
    "*": BinaryOp.MUL,
    "&": BinaryOp.AND,
    "|": BinaryOp.OR,
    "^": BinaryOp.XOR,
    "<<": BinaryOp.SHL,
    ">>": BinaryOp.SHR,
    "==": BinaryOp.EQ,
    "!=": BinaryOp.NE,
    "<": BinaryOp.LT,
    "<=": BinaryOp.LE,
    ">": BinaryOp.GT,
    ">=": BinaryOp.GE,
}

_UNOP_MAP: dict[str, UnaryOp] = {
    "~": UnaryOp.NOT,
    "b": UnaryOp.BOOL,
    "r|": UnaryOp.RED_OR,
    "r&": UnaryOp.RED_AND,
    "r^": UnaryOp.RED_XOR,
}


class _SignalNames:
    """Map Amaranth Signal objects to unique IR signal names.

    Amaranth signals are identity-based (not hashable). Two signals can
    share the same ``.name``.  We disambiguate by appending ``$N`` when
    a collision is detected.
    """

    def __init__(self):
        self._id_to_name: dict[int, str] = {}
        self._used_names: dict[str, int] = {}  # name → count

    def get(self, sig: ASignal) -> str:
        sid = id(sig)
        if sid in self._id_to_name:
            return self._id_to_name[sid]
        base = sig.name
        count = self._used_names.get(base, 0)
        if count == 0:
            name = base
        else:
            name = f"{base}${count}"
        self._used_names[base] = count + 1
        self._id_to_name[sid] = name
        return name

    def items(self) -> list[tuple[int, str]]:
        return list(self._id_to_name.items())


def _shape(val) -> Shape:
    """Convert an Amaranth shape to an IR Shape."""
    s = val.shape()
    return Shape(width=s.width, signed=s.signed)


def _lower_expr(val, names: _SignalNames, domain_signals: dict[str, tuple[str, str]]) -> Expr:
    """Lower an Amaranth Value AST node to a dau-sim IR Expr.

    Parameters
    ----------
    val : amaranth Value
        The Amaranth expression to lower.
    names : _SignalNames
        Signal name registry for unique naming.
    domain_signals : dict
        Maps domain name → (clk_signal_name, rst_signal_name) for
        resolving ClockSignal / ResetSignal references.
    """
    if isinstance(val, ASignal):
        name = names.get(val)
        return SignalRef(shape=_shape(val), name=name)

    if isinstance(val, AConst):
        return Const(shape=_shape(val), value=val.value)

    if isinstance(val, Operator):
        op_str = val.operator
        operands = val.operands
        shape = _shape(val)

        if len(operands) == 2:
            ir_op = _BINOP_MAP.get(op_str)
            if ir_op is not None:
                left = _lower_expr(operands[0], names, domain_signals)
                right = _lower_expr(operands[1], names, domain_signals)
                return Binary(shape=shape, op=ir_op, left=left, right=right)

        if len(operands) == 1:
            # Unary minus (negate)
            if op_str == "-":
                inner = _lower_expr(operands[0], names, domain_signals)
                return Unary(shape=shape, op=UnaryOp.NEG, operand=inner)

            # Sign cast operators – identity in IR (width/signedness in shape)
            if op_str in ("s", "u"):
                return _lower_expr(operands[0], names, domain_signals)

            ir_op = _UNOP_MAP.get(op_str)
            if ir_op is not None:
                inner = _lower_expr(operands[0], names, domain_signals)
                result = Unary(shape=shape, op=ir_op, operand=inner)
                if op_str == "b":
                    # bool: reduce-or then wrap as 1-bit
                    return result
                return result

        raise NotImplementedError(f"Unsupported Amaranth operator: {op_str!r} with {len(operands)} operands")

    if isinstance(val, ASlice):
        inner = _lower_expr(val.value, names, domain_signals)
        return Slice(shape=_shape(val), value=inner, low=val.start, high=val.stop)

    if isinstance(val, AConcat):
        # Amaranth Cat: parts[0] is LSB
        parts = tuple(_lower_expr(p, names, domain_signals) for p in val.parts)
        # Our IR Concat stores MSB first, Amaranth stores LSB first → reverse
        return Concat(shape=_shape(val), parts=parts[::-1])

    if isinstance(val, SwitchValue):
        # Mux(sel, if_true, if_false) → SwitchValue with 2 cases
        # cases: ((patterns, val), ...) — patterns=('0',) for false, None for default (true)
        shape = _shape(val)
        sel = _lower_expr(val.test, names, domain_signals)
        # Find the if-false (pattern='0') and if-true (default/None) branches
        if_true = None
        if_false = None
        for entry in val.cases:
            patterns = entry[0]
            v = entry[1]
            if patterns is None:
                if_true = _lower_expr(v, names, domain_signals)
            else:
                if_false = _lower_expr(v, names, domain_signals)
        if if_true is None or if_false is None:
            raise ValueError("Malformed SwitchValue/Mux")
        return Mux(shape=shape, sel=sel, if_true=if_true, if_false=if_false)

    if isinstance(val, Part):
        raise NotImplementedError("Dynamic bit/word select (Part) not yet supported")

    if isinstance(val, ClockSignal):
        domain = val.domain
        if domain in domain_signals:
            clk_name = domain_signals[domain][0]
            return SignalRef(shape=Shape(1, False), name=clk_name)
        raise ValueError(f"ClockSignal references unknown domain {domain!r}")

    if isinstance(val, ResetSignal):
        domain = val.domain
        if domain in domain_signals:
            rst_name = domain_signals[domain][1]
            return SignalRef(shape=Shape(1, False), name=rst_name)
        raise ValueError(f"ResetSignal references unknown domain {domain!r}")

    if isinstance(val, Initial):
        # Initial signal is 1 at sim start, 0 after — treat as constant 0 for RTL
        return Const(shape=Shape(1, False), value=0)

    raise NotImplementedError(f"Unsupported Amaranth value: {type(val).__name__}")


def _lower_stmts(stmts, names: _SignalNames, domain_signals: dict[str, tuple[str, str]]) -> list[Stmt]:
    """Lower a sequence of Amaranth statements to IR statements."""
    result: list[Stmt] = []
    for stmt in stmts:
        result.extend(_lower_stmt(stmt, names, domain_signals))
    return result


def _lower_stmt(stmt, names: _SignalNames, domain_signals: dict[str, tuple[str, str]]) -> list[Stmt]:
    """Lower a single Amaranth statement to IR statement(s)."""
    if isinstance(stmt, AAssign):
        target_name = _lower_lhs(stmt.lhs, names)
        value = _lower_expr(stmt.rhs, names, domain_signals)
        return [Assign(target=target_name, value=value)]

    if isinstance(stmt, ASwitch):
        test = _lower_expr(stmt.test, names, domain_signals)
        ir_cases: list[tuple[int | None, tuple[Stmt, ...]]] = []
        for entry in stmt.cases:
            patterns = entry[0]
            body = entry[1]
            ir_body = tuple(_lower_stmts(body, names, domain_signals))
            if patterns is None:
                ir_cases.append((None, ir_body))
            else:
                # Amaranth patterns are tuples of bit-strings like ('01',)
                # Each pattern is a binary string; we convert to int
                for p in patterns:
                    ir_cases.append((int(p, 2), ir_body))
        return [Switch(test=test, cases=tuple(ir_cases))]

    raise NotImplementedError(f"Unsupported Amaranth statement: {type(stmt).__name__}")


def _lower_lhs(val, names: _SignalNames) -> str:
    """Extract the target signal name from an LHS value."""
    if isinstance(val, ASignal):
        return names.get(val)
    if isinstance(val, ASlice):
        # Slice assignment: we need to handle this at the Assign level
        # For now, treat as assigning to the base signal (width handled by IR)
        return _lower_lhs(val.value, names)
    raise NotImplementedError(f"Unsupported LHS: {type(val).__name__}")


def _collect_signals_from_value(val, signals: dict[int, ASignal]):
    """Recursively collect all Signal objects from a Value AST."""
    if isinstance(val, ASignal):
        signals[id(val)] = val
    elif isinstance(val, AConst):
        pass
    elif isinstance(val, Operator):
        for op in val.operands:
            _collect_signals_from_value(op, signals)
    elif isinstance(val, ASlice):
        _collect_signals_from_value(val.value, signals)
    elif isinstance(val, AConcat):
        for p in val.parts:
            _collect_signals_from_value(p, signals)
    elif isinstance(val, SwitchValue):
        _collect_signals_from_value(val.test, signals)
        for entry in val.cases:
            _collect_signals_from_value(entry[1], signals)
    elif isinstance(val, Part):
        _collect_signals_from_value(val.value, signals)
        _collect_signals_from_value(val.offset, signals)
    elif isinstance(val, (ClockSignal, ResetSignal, Initial)):
        pass


def _collect_all_signals(frag: Fragment) -> dict[int, ASignal]:
    """Walk all statements in a fragment and collect referenced signals."""
    signals: dict[int, ASignal] = {}
    for _domain, stmts in frag.statements.items():
        for stmt in stmts:
            _collect_signals_from_stmt(stmt, signals)
    return signals


def _collect_signals_from_stmt(stmt, signals: dict[int, ASignal]):
    if isinstance(stmt, AAssign):
        _collect_signals_from_value(stmt.lhs, signals)
        _collect_signals_from_value(stmt.rhs, signals)
    elif isinstance(stmt, ASwitch):
        _collect_signals_from_value(stmt.test, signals)
        for entry in stmt.cases:
            for s in entry[1]:
                _collect_signals_from_stmt(s, signals)


def _lower_fragment(
    frag: Fragment,
    mod_name: str,
    port_map: dict[int, PortDirection] | None = None,
) -> Module:
    """Lower a single Amaranth Fragment to a dau-sim IR Module."""
    names = _SignalNames()
    port_map = port_map or {}

    # Build domain signal name mapping
    domain_signals: dict[str, tuple[str, str]] = {}
    ir_clock_domains: list[ClockDomain] = []

    # Collect all domain names: explicit (frag.domains) + implicit (statement keys)
    all_domain_names: set[str] = set(frag.iter_domains())
    for d_name in frag.statements:
        if d_name != "comb":
            all_domain_names.add(d_name)

    for d_name in sorted(all_domain_names):
        if d_name in frag.domains:
            dom = frag.domains[d_name]
            clk_name = names.get(dom.clk)
            rst_name = names.get(dom.rst)
            edge = EdgePolarity.POSEDGE if dom.clk_edge == "pos" else EdgePolarity.NEGEDGE
            rst_style = ResetStyle.ASYNC if dom.async_reset else ResetStyle.SYNC
        else:
            # Auto-create default clock domain for implicit references
            # Amaranth convention: "sync" → clk="clk", rst="rst"
            #                      other  → clk="{name}_clk", rst="{name}_rst"
            if d_name == "sync":
                clk_name, rst_name = "clk", "rst"
            else:
                clk_name, rst_name = f"{d_name}_clk", f"{d_name}_rst"
            edge = EdgePolarity.POSEDGE
            rst_style = ResetStyle.SYNC

        domain_signals[d_name] = (clk_name, rst_name)

        ir_clock_domains.append(
            ClockDomain(
                name=d_name,
                clk=clk_name,
                edge=edge,
                rst=rst_name,
                rst_style=rst_style,
                rst_active_high=True,  # Amaranth reset is always active-high
            )
        )

    # Collect all signals referenced in statements
    all_signals = _collect_all_signals(frag)
    # Register them in names
    for sid, sig in all_signals.items():
        names.get(sig)

    # Also register domain clk/rst (already done above for explicit domains)
    for d_name in frag.iter_domains():
        dom = frag.domains[d_name]
        all_signals[id(dom.clk)] = dom.clk
        all_signals[id(dom.rst)] = dom.rst

    # Lower statements per domain
    comb_blocks: list[CombBlock] = []
    seq_blocks: list[SeqBlock] = []

    for domain, stmts in frag.statements.items():
        ir_stmts = _lower_stmts(stmts, names, domain_signals)
        if not ir_stmts:
            continue
        if domain == "comb":
            comb_blocks.append(CombBlock(stmts=tuple(ir_stmts)))
        else:
            seq_blocks.append(SeqBlock(domain=domain, stmts=tuple(ir_stmts)))

    # Build ports and signals
    ports: list[Port] = []
    signals: list[Signal] = []
    seen_names: set[str] = set()

    for sid, asig in all_signals.items():
        ir_name = names.get(asig)
        if ir_name in seen_names:
            continue
        seen_names.add(ir_name)
        ir_shape = Shape(width=asig.shape().width, signed=asig.shape().signed)
        ir_sig = Signal(name=ir_name, shape=ir_shape, init=asig.init)

        if sid in port_map:
            ports.append(Port(signal=ir_sig, direction=port_map[sid]))
        else:
            signals.append(ir_sig)

    # Add clock/reset as ports (inputs) — both explicit and implicit domains
    for d_name in sorted(all_domain_names):
        clk_name, rst_name = domain_signals[d_name]
        for sig_name in (clk_name, rst_name):
            if sig_name not in seen_names:
                seen_names.add(sig_name)
                ir_sig = Signal(name=sig_name, shape=Shape(1, False))
                ports.append(Port(signal=ir_sig, direction=PortDirection.INPUT))

    return Module(
        name=mod_name,
        ports=tuple(ports),
        signals=tuple(signals),
        clock_domains=tuple(ir_clock_domains),
        comb_blocks=tuple(comb_blocks),
        seq_blocks=tuple(seq_blocks),
    )


def from_amaranth(
    design: Elaboratable,
    *,
    name: str | None = None,
    platform=None,
) -> Module:
    """Lower an Amaranth ``Elaboratable`` or ``Component`` to a dau-sim IR Module.

    Parameters
    ----------
    design : Elaboratable
        The Amaranth design to elaborate and lower.  Can be a bare
        ``Elaboratable`` or a ``Component`` (which provides port info
        via its signature).
    name : str | None
        Module name. Defaults to the class name of *design*.
    platform : object | None
        Amaranth platform, passed to ``Fragment.get()``.

    Returns
    -------
    Module
        The lowered dau-sim IR module.
    """
    mod_name = name or type(design).__name__

    frag = Fragment.get(design, platform=platform)

    # Build port map from Component signature if available
    port_map: dict[int, PortDirection] = {}
    try:
        from amaranth.lib.wiring import Component

        if isinstance(design, Component):
            for member_name, member in design.signature.members.items():
                sig = getattr(design, member_name)
                if isinstance(sig, ASignal):
                    flow_str = str(member.flow)
                    if "In" in flow_str:
                        port_map[id(sig)] = PortDirection.INPUT
                    elif "Out" in flow_str:
                        port_map[id(sig)] = PortDirection.OUTPUT
    except ImportError:
        pass

    return _lower_fragment(frag, mod_name, port_map)
