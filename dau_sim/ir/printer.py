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
from dau_sim.ir.module import (
    Module,
    Port,
    Signal,
)
from dau_sim.ir.stmt import Assert, Assign, IfElse, Print, Stmt, Switch
from dau_sim.ir.types import PortDirection

_UNARY_SYMBOLS: dict[UnaryOp, str] = {
    UnaryOp.NOT: "~",
    UnaryOp.NEG: "-",
    UnaryOp.BOOL: "bool",
    UnaryOp.RED_AND: "&",
    UnaryOp.RED_OR: "|",
    UnaryOp.RED_XOR: "^",
}

_BINARY_SYMBOLS: dict[BinaryOp, str] = {
    BinaryOp.ADD: "+",
    BinaryOp.SUB: "-",
    BinaryOp.MUL: "*",
    BinaryOp.DIV: "/",
    BinaryOp.MOD: "%",
    BinaryOp.AND: "&",
    BinaryOp.OR: "|",
    BinaryOp.XOR: "^",
    BinaryOp.SHL: "<<",
    BinaryOp.SHR: ">>",
    BinaryOp.EQ: "==",
    BinaryOp.NE: "!=",
    BinaryOp.LT: "<",
    BinaryOp.LE: "<=",
    BinaryOp.GT: ">",
    BinaryOp.GE: ">=",
    BinaryOp.LOGIC_AND: "&&",
    BinaryOp.LOGIC_OR: "||",
}

_DIR_SYMBOLS: dict[PortDirection, str] = {
    PortDirection.INPUT: "input",
    PortDirection.OUTPUT: "output",
    PortDirection.INOUT: "inout",
}


def fmt_expr(e: Expr) -> str:
    """Format an expression as a compact string."""
    if isinstance(e, Const):
        return repr(e)
    if isinstance(e, SignalRef):
        return e.name
    if isinstance(e, Unary):
        sym = _UNARY_SYMBOLS[e.op]
        if e.op == UnaryOp.BOOL:
            return f"bool({fmt_expr(e.operand)})"
        return f"({sym}{fmt_expr(e.operand)})"
    if isinstance(e, Binary):
        sym = _BINARY_SYMBOLS[e.op]
        return f"({fmt_expr(e.left)} {sym} {fmt_expr(e.right)})"
    if isinstance(e, Mux):
        return f"({fmt_expr(e.sel)} ? {fmt_expr(e.if_true)} : {fmt_expr(e.if_false)})"
    if isinstance(e, Concat):
        parts = ", ".join(fmt_expr(p) for p in e.parts)
        return f"{{{parts}}}"
    if isinstance(e, Slice):
        return f"{fmt_expr(e.value)}[{e.low}:{e.high}]"
    return f"<unknown expr {type(e).__name__}>"


def fmt_stmt(s: Stmt, indent: int = 0) -> str:
    """Format a statement with indentation."""
    pad = "  " * indent
    if isinstance(s, Assign):
        return f"{pad}{s.target} = {fmt_expr(s.value)}"
    if isinstance(s, IfElse):
        lines = [f"{pad}if {fmt_expr(s.cond)}:"]
        for st in s.then_body:
            lines.append(fmt_stmt(st, indent + 1))
        if s.else_body:
            lines.append(f"{pad}else:")
            for st in s.else_body:
                lines.append(fmt_stmt(st, indent + 1))
        return "\n".join(lines)
    if isinstance(s, Switch):
        lines = [f"{pad}switch {fmt_expr(s.test)}:"]
        for pattern, stmts in s.cases:
            label = "default" if pattern is None else str(pattern)
            lines.append(f"{pad}  case {label}:")
            for st in stmts:
                lines.append(fmt_stmt(st, indent + 2))
        return "\n".join(lines)
    if isinstance(s, Assert):
        msg = f' "{s.message}"' if s.message else ""
        return f"{pad}assert {fmt_expr(s.cond)}{msg}"
    if isinstance(s, Print):
        args = ", ".join(fmt_expr(a) for a in s.args)
        return f'{pad}$display("{s.format_str}", {args})'
    return f"{pad}<unknown stmt {type(s).__name__}>"


def fmt_signal(s: Signal) -> str:
    """Format a signal declaration."""
    init = f" = {s.init}" if s.init != 0 else ""
    return f"{s.shape} {s.name}{init}"


def fmt_port(p: Port) -> str:
    """Format a port declaration."""
    d = _DIR_SYMBOLS[p.direction]
    init = f" = {p.signal.init}" if p.signal.init != 0 else ""
    return f"{d} {p.shape} {p.name}{init}"


def fmt_module(m: Module) -> str:
    """Format a complete module as a readable string."""
    lines: list[str] = []
    lines.append(f"module {m.name}:")

    if m.ports:
        lines.append("  ports:")
        for p in m.ports:
            lines.append(f"    {fmt_port(p)}")

    if m.signals:
        lines.append("  signals:")
        for s in m.signals:
            lines.append(f"    {fmt_signal(s)}")

    if m.clock_domains:
        lines.append("  clock domains:")
        for cd in m.clock_domains:
            rst = f", rst={cd.rst} ({cd.rst_style.name})" if cd.rst else ""
            lines.append(f"    {cd.name}: clk={cd.clk} ({cd.edge.name}){rst}")

    if m.comb_blocks:
        for i, cb in enumerate(m.comb_blocks):
            lines.append(f"  comb block {i}:")
            for s in cb.stmts:
                lines.append(fmt_stmt(s, indent=2))

    if m.seq_blocks:
        for i, sb in enumerate(m.seq_blocks):
            lines.append(f"  seq block {i} (domain={sb.domain}):")
            for s in sb.stmts:
                lines.append(fmt_stmt(s, indent=2))

    if m.init_blocks:
        for i, ib in enumerate(m.init_blocks):
            lines.append(f"  init block {i}:")
            for s in ib.stmts:
                lines.append(fmt_stmt(s, indent=2))

    if m.instances:
        lines.append("  instances:")
        for inst in m.instances:
            params = ""
            if inst.parameters:
                ps = ", ".join(f"{k}={v}" for k, v in inst.parameters.items())
                params = f" #({ps})"
            lines.append(f"    {inst.module_name}{params} {inst.name}")
            for b in inst.bindings:
                lines.append(f"      .{b.port_name}({fmt_expr(b.expr)})")

    if m.memories:
        lines.append("  memories:")
        for mem in m.memories:
            lines.append(f"    {mem.name}: {mem.shape} x {mem.depth}")

    return "\n".join(lines)
