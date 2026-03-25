"""Signal name prefix rewriter for hierarchy flattening.

Walks IR expression and statement trees, prepending a hierarchical prefix
to every signal name reference.  Used by the flattener to scope child
module signals under their instance name (e.g., ``count`` → ``u0.count``).
"""

from __future__ import annotations

from dau_sim.ir.expr import Binary, Concat, Const, Expr, Mux, SignalRef, Slice, Unary
from dau_sim.ir.stmt import Assert, Assign, Delay, Finish, IfElse, Print, Stmt, Switch


def prefix_expr(expr: Expr, prefix: str) -> Expr:
    """Return a copy of *expr* with every ``SignalRef.name`` prefixed."""
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, SignalRef):
        return SignalRef(shape=expr.shape, name=f"{prefix}.{expr.name}")
    if isinstance(expr, Unary):
        return Unary(shape=expr.shape, op=expr.op, operand=prefix_expr(expr.operand, prefix))
    if isinstance(expr, Binary):
        return Binary(
            shape=expr.shape,
            op=expr.op,
            left=prefix_expr(expr.left, prefix),
            right=prefix_expr(expr.right, prefix),
        )
    if isinstance(expr, Mux):
        return Mux(
            shape=expr.shape,
            sel=prefix_expr(expr.sel, prefix),
            if_true=prefix_expr(expr.if_true, prefix),
            if_false=prefix_expr(expr.if_false, prefix),
        )
    if isinstance(expr, Concat):
        return Concat(
            shape=expr.shape,
            parts=tuple(prefix_expr(p, prefix) for p in expr.parts),
        )
    if isinstance(expr, Slice):
        return Slice(
            shape=expr.shape,
            value=prefix_expr(expr.value, prefix),
            low=expr.low,
            high=expr.high,
        )
    raise NotImplementedError(f"prefix_expr: unsupported {type(expr).__name__}")


def prefix_stmt(stmt: Stmt, prefix: str) -> Stmt:
    """Return a copy of *stmt* with every signal reference prefixed."""
    if isinstance(stmt, Assign):
        return Assign(
            target=f"{prefix}.{stmt.target}",
            value=prefix_expr(stmt.value, prefix),
        )
    if isinstance(stmt, IfElse):
        return IfElse(
            cond=prefix_expr(stmt.cond, prefix),
            then_body=tuple(prefix_stmt(s, prefix) for s in stmt.then_body),
            else_body=tuple(prefix_stmt(s, prefix) for s in stmt.else_body),
        )
    if isinstance(stmt, Switch):
        return Switch(
            test=prefix_expr(stmt.test, prefix),
            cases=tuple((pat, tuple(prefix_stmt(s, prefix) for s in body)) for pat, body in stmt.cases),
        )
    if isinstance(stmt, Assert):
        return Assert(
            cond=prefix_expr(stmt.cond, prefix),
            message=stmt.message,
        )
    if isinstance(stmt, Print):
        return Print(
            format_str=stmt.format_str,
            args=tuple(prefix_expr(a, prefix) for a in stmt.args),
        )
    if isinstance(stmt, (Delay, Finish)):
        return stmt
    raise NotImplementedError(f"prefix_stmt: unsupported {type(stmt).__name__}")


def prefix_stmts(stmts: tuple[Stmt, ...], prefix: str) -> tuple[Stmt, ...]:
    """Prefix all signal references in a tuple of statements."""
    return tuple(prefix_stmt(s, prefix) for s in stmts)
