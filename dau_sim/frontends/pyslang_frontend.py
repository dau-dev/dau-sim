from __future__ import annotations

import pyslang as ps

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
from dau_sim.ir.stmt import Assign, IfElse, Stmt
from dau_sim.ir.types import EdgePolarity, PortDirection, ResetStyle, Shape

_ps_ast = getattr(ps, "ast", ps)
_ps_syntax = getattr(ps, "syntax", ps)

__all__ = (
    "parse_sv",
    "parse_sv_file",
    "from_dau_build",
)

_BINOP_MAP: dict[_ps_ast.BinaryOperator, BinaryOp] = {
    _ps_ast.BinaryOperator.Add: BinaryOp.ADD,
    _ps_ast.BinaryOperator.Subtract: BinaryOp.SUB,
    _ps_ast.BinaryOperator.Multiply: BinaryOp.MUL,
    _ps_ast.BinaryOperator.Divide: BinaryOp.DIV,
    _ps_ast.BinaryOperator.Mod: BinaryOp.MOD,
    _ps_ast.BinaryOperator.BinaryAnd: BinaryOp.AND,
    _ps_ast.BinaryOperator.BinaryOr: BinaryOp.OR,
    _ps_ast.BinaryOperator.BinaryXor: BinaryOp.XOR,
    _ps_ast.BinaryOperator.LogicalShiftLeft: BinaryOp.SHL,
    _ps_ast.BinaryOperator.LogicalShiftRight: BinaryOp.SHR,
    _ps_ast.BinaryOperator.ArithmeticShiftLeft: BinaryOp.SHL,
    _ps_ast.BinaryOperator.ArithmeticShiftRight: BinaryOp.SHR,
    _ps_ast.BinaryOperator.Equality: BinaryOp.EQ,
    _ps_ast.BinaryOperator.Inequality: BinaryOp.NE,
    _ps_ast.BinaryOperator.CaseEquality: BinaryOp.EQ,
    _ps_ast.BinaryOperator.CaseInequality: BinaryOp.NE,
    _ps_ast.BinaryOperator.LessThan: BinaryOp.LT,
    _ps_ast.BinaryOperator.LessThanEqual: BinaryOp.LE,
    _ps_ast.BinaryOperator.GreaterThan: BinaryOp.GT,
    _ps_ast.BinaryOperator.GreaterThanEqual: BinaryOp.GE,
    _ps_ast.BinaryOperator.LogicalAnd: BinaryOp.LOGIC_AND,
    _ps_ast.BinaryOperator.LogicalOr: BinaryOp.LOGIC_OR,
}

_UNOP_MAP: dict[_ps_ast.UnaryOperator, UnaryOp] = {
    _ps_ast.UnaryOperator.BitwiseNot: UnaryOp.NOT,
    _ps_ast.UnaryOperator.Minus: UnaryOp.NEG,
    _ps_ast.UnaryOperator.Plus: None,  # unary + is identity
    _ps_ast.UnaryOperator.LogicalNot: UnaryOp.BOOL,  # !x == ~(|x)
    _ps_ast.UnaryOperator.BitwiseAnd: UnaryOp.RED_AND,
    _ps_ast.UnaryOperator.BitwiseNand: UnaryOp.RED_AND,  # post-negate
    _ps_ast.UnaryOperator.BitwiseOr: UnaryOp.RED_OR,
    _ps_ast.UnaryOperator.BitwiseNor: UnaryOp.RED_OR,  # post-negate
    _ps_ast.UnaryOperator.BitwiseXor: UnaryOp.RED_XOR,
    _ps_ast.UnaryOperator.BitwiseXnor: UnaryOp.RED_XOR,  # post-negate
}

# Operators that need a post-inversion after the reduction
_UNOP_NEGATE: set[_ps_ast.UnaryOperator] = {
    _ps_ast.UnaryOperator.LogicalNot,
    _ps_ast.UnaryOperator.BitwiseNand,
    _ps_ast.UnaryOperator.BitwiseNor,
    _ps_ast.UnaryOperator.BitwiseXnor,
}


def _shape_of(expr) -> Shape:
    """Extract an IR Shape from a pyslang expression's type."""
    t = expr.type
    return Shape(width=t.bitWidth, signed=t.isSigned)


def _lower_expr(expr) -> Expr:
    """Lower a pyslang expression AST node to a dau-sim IR Expr."""
    kind = expr.kind

    # Unwrap implicit conversions (sign/width extension)
    if kind == _ps_ast.ExpressionKind.Conversion:
        inner = _lower_expr(expr.operand)
        target_shape = _shape_of(expr)
        # If shapes differ, the IR evaluator handles width via mask_value;
        # we just adjust the shape wrapper.
        if inner.shape == target_shape:
            return inner
        # Re-wrap with target shape — the evaluator's mask_value handles truncation/extension
        return _rewrap(inner, target_shape)

    if kind == _ps_ast.ExpressionKind.IntegerLiteral:
        return Const(shape=_shape_of(expr), value=int(expr.value))

    if kind == _ps_ast.ExpressionKind.NamedValue:
        if expr.symbol.kind == _ps_ast.SymbolKind.Parameter:
            return Const(shape=_shape_of(expr), value=_constant_value_to_int(expr.symbol.value))
        return SignalRef(shape=_shape_of(expr), name=expr.symbol.name)

    if kind == _ps_ast.ExpressionKind.BinaryOp:
        op = _BINOP_MAP.get(expr.op)
        if op is None:
            raise NotImplementedError(f"Unsupported binary operator: {expr.op}")
        left = _lower_expr(expr.left)
        right = _lower_expr(expr.right)
        return Binary(shape=_shape_of(expr), op=op, left=left, right=right)

    if kind == _ps_ast.ExpressionKind.UnaryOp:
        mapped = _UNOP_MAP.get(expr.op)
        if mapped is None and expr.op != _ps_ast.UnaryOperator.Plus:
            raise NotImplementedError(f"Unsupported unary operator: {expr.op}")
        operand = _lower_expr(expr.operand)
        if expr.op == _ps_ast.UnaryOperator.Plus:
            return operand  # identity
        result = Unary(shape=_shape_of(expr), op=mapped, operand=operand)
        if expr.op in _UNOP_NEGATE:
            # LogicalNot: !x means ~(|x) → BOOL then NOT
            # NAND/NOR/XNOR: reduction then NOT
            if expr.op == _ps_ast.UnaryOperator.LogicalNot:
                # BOOL already gives 1-bit result; we need to invert it
                return Unary(shape=Shape(1, False), op=UnaryOp.NOT, operand=result)
            else:
                return Unary(shape=Shape(1, False), op=UnaryOp.NOT, operand=result)
        return result

    if kind == _ps_ast.ExpressionKind.ConditionalOp:
        # Ternary: sel ? left : right
        cond_expr = expr.conditions[0].expr
        sel = _lower_expr(cond_expr)
        if_true = _lower_expr(expr.left)
        if_false = _lower_expr(expr.right)
        return Mux(shape=_shape_of(expr), sel=sel, if_true=if_true, if_false=if_false)

    if kind == _ps_ast.ExpressionKind.Concatenation:
        parts = tuple(_lower_expr(op) for op in expr.operands)
        return Concat(shape=_shape_of(expr), parts=parts)

    if kind == _ps_ast.ExpressionKind.RangeSelect:
        value = _lower_expr(expr.value)
        high_bit = int(expr.left.value)  # SV: a[high:low]
        low_bit = int(expr.right.value)
        return Slice(
            shape=_shape_of(expr),
            value=value,
            low=low_bit,
            high=high_bit + 1,  # IR uses exclusive upper bound
        )

    if kind == _ps_ast.ExpressionKind.ElementSelect:
        value = _lower_expr(expr.value)
        idx = _lower_expr(expr.selector)
        # Single-bit select: a[i] → Slice(low=i, high=i+1)
        if idx.shape.width <= 32 and isinstance(idx, Const):
            return Slice(
                shape=Shape(1, False),
                value=value,
                low=idx.value,
                high=idx.value + 1,
            )
        raise NotImplementedError("Dynamic bit select not yet supported")

    if kind == _ps_ast.ExpressionKind.Replication:
        # {N{expr}} — replicate expr N times → Concat
        inner = _lower_expr(expr.operands[0]) if hasattr(expr, "operands") else _lower_expr(expr.concat)
        count = _shape_of(expr).width // inner.shape.width
        parts = tuple(inner for _ in range(count))
        return Concat(shape=_shape_of(expr), parts=parts)

    if kind == _ps_ast.ExpressionKind.Assignment:
        # This shouldn't appear as an expression we need to lower to Expr;
        # assignments are handled at the statement level.
        raise ValueError("Assignment expression encountered in expression context")

    raise NotImplementedError(f"Unsupported expression kind: {kind} ({type(expr).__name__})")


def _rewrap(inner: Expr, target: Shape) -> Expr:
    """Adjust an expression's width/signedness to match a target shape.

    Uses Slice for truncation (wider→narrower). For extension (narrower→wider),
    the evaluator's mask_value handles it — we just re-create the node with the
    target shape.
    """
    if inner.shape == target:
        return inner
    if inner.shape.width > target.width:
        return Slice(shape=target, value=inner, low=0, high=target.width)
    # Extension: wrap in a Concat with zero-padding MSB bits
    pad_width = target.width - inner.shape.width
    if pad_width > 0:
        pad = Const(shape=Shape(pad_width, False), value=0)
        return Concat(shape=target, parts=(pad, inner))
    # Same width, different signedness — just return inner; evaluator handles it
    return inner


def _constant_value_to_int(value) -> int:
    if value.hasUnknown():
        raise ValueError("parameter constants with unknown bits are not supported")
    return int(value.value.toString(ps.LiteralBase.Decimal, False).replace("_", ""))


def _flatten_stmts(stmt) -> list[Stmt]:
    """Recursively lower a pyslang statement AST node to a list of IR Stmts."""
    kind = stmt.kind

    if kind == _ps_ast.StatementKind.ExpressionStatement:
        return _lower_assign_stmt(stmt.expr)

    if kind == _ps_ast.StatementKind.Block:
        body = stmt.body
        result: list[Stmt] = []
        if body.kind == _ps_ast.StatementKind.List:
            for s in body.list:
                result.extend(_flatten_stmts(s))
        else:
            result.extend(_flatten_stmts(body))
        return result

    if kind == _ps_ast.StatementKind.List:
        result = []
        for s in stmt.list:
            result.extend(_flatten_stmts(s))
        return result

    if kind == _ps_ast.StatementKind.Conditional:
        cond_expr = _lower_expr(stmt.conditions[0].expr)
        then_body = tuple(_flatten_stmts(stmt.ifTrue))
        else_body = tuple(_flatten_stmts(stmt.ifFalse)) if stmt.ifFalse else ()
        return [IfElse(cond=cond_expr, then_body=then_body, else_body=else_body)]

    raise NotImplementedError(f"Unsupported statement kind: {kind} ({type(stmt).__name__})")


def _lower_assign_stmt(expr) -> list[Stmt]:
    """Lower an AssignmentExpression to IR Assign statement(s).

    Handles simple named targets and LHS concatenation (e.g. {carry, sum} = ...).
    """
    if expr.kind != _ps_ast.ExpressionKind.Assignment:
        raise NotImplementedError(f"Expected assignment expression, got {expr.kind}")
    lhs = expr.left
    rhs = _lower_expr(expr.right)

    if lhs.kind == _ps_ast.ExpressionKind.NamedValue:
        return [Assign(target=lhs.symbol.name, value=rhs)]

    if lhs.kind == _ps_ast.ExpressionKind.Concatenation:
        # Split {a, b, c} = rhs into:
        #   a = rhs[total-1 : total-a.width]
        #   b = rhs[total-a.width-1 : total-a.width-b.width]
        #   etc.
        stmts: list[Stmt] = []
        total_width = lhs.type.bitWidth
        offset = total_width
        for operand in lhs.operands:
            if operand.kind != _ps_ast.ExpressionKind.NamedValue:
                raise NotImplementedError(f"Unsupported LHS concat operand: {operand.kind}")
            w = operand.type.bitWidth
            offset -= w
            slice_expr = Slice(
                shape=Shape(w, operand.type.isSigned),
                value=rhs,
                low=offset,
                high=offset + w,
            )
            stmts.append(Assign(target=operand.symbol.name, value=slice_expr))
        return stmts

    raise NotImplementedError(f"Unsupported LHS expression kind: {lhs.kind}")


def _extract_timing(timed_stmt) -> tuple[list[tuple[str, EdgePolarity]], object]:
    """Extract clock/reset edge info from a TimedStatement.

    Returns (events, inner_stmt) where events is a list of (signal_name, edge).
    """
    timing = timed_stmt.timing
    events = []

    if hasattr(timing, "events"):
        # EventListControl: always_ff @(posedge clk or negedge rst)
        for ev in timing.events:
            edge = _map_edge(ev.edge)
            sig_name = ev.expr.symbol.name if hasattr(ev.expr, "symbol") else None
            if sig_name:
                events.append((sig_name, edge))
    elif hasattr(timing, "edge"):
        # SignalEventControl: always @(posedge clk)
        edge = _map_edge(timing.edge)
        sig_name = timing.expr.symbol.name if hasattr(timing.expr, "symbol") else None
        if sig_name:
            events.append((sig_name, edge))

    return events, timed_stmt.stmt


def _map_edge(edge_kind) -> EdgePolarity:
    if edge_kind == _ps_ast.EdgeKind.PosEdge:
        return EdgePolarity.POSEDGE
    elif edge_kind == _ps_ast.EdgeKind.NegEdge:
        return EdgePolarity.NEGEDGE
    else:
        return EdgePolarity.BOTH


def _infer_reset_from_events(
    events: list[tuple[str, EdgePolarity]],
) -> tuple[str | None, ResetStyle, bool]:
    """Infer reset signal, style, and polarity from the event list.

    For async reset: the event list has 2+ entries; the first non-clock
    event is taken as the reset.
    """
    if len(events) < 2:
        return None, ResetStyle.SYNC, True

    # The non-clock events are potential async resets
    reset_candidates = [(sig, edge) for sig, edge in events[1:]]

    if not reset_candidates:
        return None, ResetStyle.SYNC, True

    rst_sig, rst_edge = reset_candidates[0]
    # Active-high if posedge, active-low if negedge
    rst_active_high = rst_edge == EdgePolarity.POSEDGE

    return rst_sig, ResetStyle.ASYNC, rst_active_high


def _lower_module_instance(inst) -> Module:
    """Lower a single pyslang module instance to a dau-sim IR Module."""
    mod_name = inst.name

    ports: list[Port] = []
    signals: list[Signal] = []
    comb_blocks: list[CombBlock] = []
    seq_blocks: list[SeqBlock] = []
    clock_domains: list[ClockDomain] = []

    # Track which signal names we've seen (ports also define signals)
    seen_signals: set[str] = set()
    # Track domain names to avoid duplicates
    domain_names: set[str] = set()

    for child in inst.body:
        sym_kind = child.kind

        if sym_kind == _ps_ast.SymbolKind.Port:
            sig_name = child.name
            shape = Shape(width=child.type.bitWidth, signed=child.type.isSigned)
            direction = _map_direction(child.direction)
            sig = Signal(name=sig_name, shape=shape)
            ports.append(Port(signal=sig, direction=direction))
            seen_signals.add(sig_name)

        elif sym_kind == _ps_ast.SymbolKind.Net:
            sig_name = child.name
            if sig_name not in seen_signals:
                shape = Shape(width=child.type.bitWidth, signed=child.type.isSigned)
                signals.append(Signal(name=sig_name, shape=shape))
                seen_signals.add(sig_name)

        elif sym_kind == _ps_ast.SymbolKind.Variable:
            sig_name = child.name
            if sig_name not in seen_signals:
                shape = Shape(width=child.type.bitWidth, signed=child.type.isSigned)
                signals.append(Signal(name=sig_name, shape=shape))
                seen_signals.add(sig_name)

        elif sym_kind == _ps_ast.SymbolKind.ContinuousAssign:
            # assign y = expr;
            assign_expr = child.assignment
            stmts = _lower_assign_stmt(assign_expr)
            comb_blocks.append(CombBlock(stmts=tuple(stmts)))

        elif sym_kind == _ps_ast.SymbolKind.ProceduralBlock:
            pk = child.procedureKind

            if pk == _ps_ast.ProceduralBlockKind.AlwaysComb:
                stmts = _flatten_stmts(child.body)
                comb_blocks.append(CombBlock(stmts=tuple(stmts)))

            elif pk in (_ps_ast.ProceduralBlockKind.AlwaysFF, _ps_ast.ProceduralBlockKind.Always):
                body = child.body

                if body.kind == _ps_ast.StatementKind.Timed:
                    events, inner_stmt = _extract_timing(body)
                    stmts = _flatten_stmts(inner_stmt)

                    if events:
                        clk_sig, clk_edge = events[0]
                        rst_sig, rst_style, rst_active_high = _infer_reset_from_events(events)

                        domain_name = clk_sig
                        # Deduplicate: if domain already exists with same config, reuse
                        if domain_name not in domain_names:
                            domain = ClockDomain(
                                name=domain_name,
                                clk=clk_sig,
                                edge=clk_edge,
                                rst=rst_sig,
                                rst_style=rst_style,
                                rst_active_high=rst_active_high,
                            )
                            clock_domains.append(domain)
                            domain_names.add(domain_name)

                        seq_blocks.append(SeqBlock(domain=domain_name, stmts=tuple(stmts)))
                    else:
                        # No events — treat as combinational
                        comb_blocks.append(CombBlock(stmts=tuple(stmts)))
                else:
                    # No timing control — treat as combinational
                    stmts = _flatten_stmts(body)
                    comb_blocks.append(CombBlock(stmts=tuple(stmts)))

    return Module(
        name=mod_name,
        ports=tuple(ports),
        signals=tuple(signals),
        clock_domains=tuple(clock_domains),
        comb_blocks=tuple(comb_blocks),
        seq_blocks=tuple(seq_blocks),
    )


def _map_direction(direction) -> PortDirection:
    if direction == _ps_ast.ArgumentDirection.In:
        return PortDirection.INPUT
    elif direction == _ps_ast.ArgumentDirection.Out:
        return PortDirection.OUTPUT
    elif direction == _ps_ast.ArgumentDirection.InOut:
        return PortDirection.INOUT
    raise ValueError(f"Unsupported port direction: {direction}")


def parse_sv(source: str, *, top: str | None = None) -> Module:
    """Parse SystemVerilog/Verilog source text and return a dau-sim IR Module.

    Parameters
    ----------
    source : str
        SystemVerilog or Verilog source code.
    top : str | None
        Name of the top-level module to extract. If *None*, uses the first
        top instance from elaboration.

    Returns
    -------
    Module
        The lowered dau-sim IR module.

    Raises
    ------
    ValueError
        If the source has compilation errors or the requested top module
        is not found.
    """
    tree = _ps_syntax.SyntaxTree.fromText(source)
    comp = _ps_ast.Compilation()
    comp.addSyntaxTree(tree)

    # Check for compilation errors
    diags = comp.getAllDiagnostics()
    errors = [d for d in diags if d.isError()]
    if errors:
        report = ps.DiagnosticEngine.reportAll(comp.sourceManager, diags)
        raise ValueError(f"Compilation errors:\n{report}")

    root = comp.getRoot()
    instances = root.topInstances

    if not instances:
        raise ValueError("No top-level module found in source")

    if top is not None:
        for inst in instances:
            if inst.name == top:
                return _lower_module_instance(inst)
        raise ValueError(f"Top module '{top}' not found. Available: {[i.name for i in instances]}")

    return _lower_module_instance(instances[0])


def parse_sv_file(path: str, *, top: str | None = None) -> Module:
    """Parse a SystemVerilog/Verilog file and return a dau-sim IR Module.

    Parameters
    ----------
    path : str
        Path to the .sv or .v file.
    top : str | None
        Name of the top-level module. If *None*, uses the first top instance.

    Returns
    -------
    Module
        The lowered dau-sim IR module.
    """
    with open(path) as f:
        source = f.read()
    return parse_sv(source, top=top)


def from_dau_build(mod, *, top: str | None = None) -> Module:
    """Bridge: lower a ``dau_build.Module`` to a dau-sim IR :class:`Module`.

    ``dau_build`` performs *syntactic* extraction (ports, wires, hierarchy as
    metadata with string expressions).  ``dau-sim`` needs *semantic* lowering
    (typed expression/statement IR for simulation).  This bridge re-compiles the
    underlying source via pyslang's Compilation API to produce a simulatable IR.

    Parameters
    ----------
    mod : dau_build.Module
        A module previously obtained via ``dau_build.Module.from_file()`` or
        ``dau_build.Module.from_str()``.  Must have a ``source_path`` pointing
        to the original ``.sv`` file, or a ``node`` from which source text can
        be recovered.
    top : str | None
        Name of the top-level module to extract.  Defaults to the name stored
        in *mod*.

    Returns
    -------
    Module
        The lowered dau-sim IR module.
    """
    top = top or mod.name

    # Recover source text
    if mod.source_path is not None:
        return parse_sv_file(str(mod.source_path), top=top)

    # Fallback: reconstruct source from the syntax node stored on the model
    if mod.node is not None:
        source = str(mod.node)
        return parse_sv(source, top=top)

    raise ValueError("Cannot lower dau_build.Module: no source_path or syntax node available.")
