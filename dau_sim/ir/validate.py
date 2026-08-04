from ccflow import BaseModel
from pydantic import ConfigDict

from dau_sim.ir.expr import (
    Binary,
    Concat,
    Const,
    Expr,
    Mux,
    SignalRef,
    Slice,
    Unary,
)
from dau_sim.ir.module import Module
from dau_sim.ir.stmt import Assert, Assign, IfElse, Print, Stmt, Switch


class ValidationError(BaseModel):
    """A single validation error."""

    model_config = ConfigDict(frozen=True)

    path: str  # e.g. "module.comb_block[0].stmt[1]"
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ValidationResult(BaseModel):
    """Collected validation errors."""

    model_config = ConfigDict(frozen=True)

    errors: tuple[ValidationError, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        if self.ok:
            return "OK"
        return "\n".join(str(e) for e in self.errors)


def validate_module(module: Module) -> ValidationResult:
    """Validate a module for well-formedness."""
    errors: list[ValidationError] = []
    prefix = f"module({module.name})"

    # Collect all known signal names
    known_signals = module.all_signal_names

    # Check for duplicate signal names
    seen: set[str] = set()
    for p in module.ports:
        if p.name in seen:
            _add_error(errors, prefix, f"duplicate signal name: {p.name}")
        seen.add(p.name)
    for s in module.signals:
        if s.name in seen:
            _add_error(errors, prefix, f"duplicate signal name: {s.name}")
        seen.add(s.name)

    # Check clock domains reference valid signals
    for cd in module.clock_domains:
        if cd.clk not in known_signals:
            _add_error(
                errors,
                f"{prefix}.clock_domain({cd.name})",
                f"clock signal '{cd.clk}' not found",
            )
        if cd.rst and cd.rst not in known_signals:
            _add_error(
                errors,
                f"{prefix}.clock_domain({cd.name})",
                f"reset signal '{cd.rst}' not found",
            )

    # Check sequentially blocks reference valid clock domains
    domain_names = {cd.name for cd in module.clock_domains}
    for i, sb in enumerate(module.seq_blocks):
        if sb.domain not in domain_names:
            _add_error(
                errors,
                f"{prefix}.seq_block[{i}]",
                f"unknown clock domain: {sb.domain}",
            )

    # Validate all blocks' statements
    for i, cb in enumerate(module.comb_blocks):
        _validate_block_stmts(cb.stmts, known_signals, f"{prefix}.comb_block[{i}]", errors)
    for i, sb in enumerate(module.seq_blocks):
        _validate_block_stmts(sb.stmts, known_signals, f"{prefix}.seq_block[{i}]", errors)
    for i, ib in enumerate(module.init_blocks):
        _validate_block_stmts(ib.stmts, known_signals, f"{prefix}.init_block[{i}]", errors)

    return ValidationResult(errors=tuple(errors))


def _add_error(errors: list[ValidationError], path: str, message: str) -> None:
    errors.append(ValidationError(path=path, message=message))


def _validate_block_stmts(
    stmts: tuple[Stmt, ...],
    known_signals: set[str],
    path: str,
    errors: list[ValidationError],
) -> None:
    """Validate statements within a block."""
    for i, stmt in enumerate(stmts):
        _validate_stmt(stmt, known_signals, f"{path}.stmt[{i}]", errors)


def _validate_stmt(
    stmt: Stmt,
    known_signals: set[str],
    path: str,
    errors: list[ValidationError],
) -> None:
    """Validate a single statement."""
    if isinstance(stmt, Assign):
        if stmt.target not in known_signals:
            _add_error(errors, path, f"assignment to unknown signal: {stmt.target}")
        _validate_expr(stmt.value, known_signals, f"{path}.value", errors)
    elif isinstance(stmt, IfElse):
        _validate_expr(stmt.cond, known_signals, f"{path}.cond", errors)
        for j, s in enumerate(stmt.then_body):
            _validate_stmt(s, known_signals, f"{path}.then[{j}]", errors)
        for j, s in enumerate(stmt.else_body):
            _validate_stmt(s, known_signals, f"{path}.else[{j}]", errors)
    elif isinstance(stmt, Switch):
        _validate_expr(stmt.test, known_signals, f"{path}.test", errors)
        for ci, (_, stmts) in enumerate(stmt.cases):
            for j, s in enumerate(stmts):
                _validate_stmt(s, known_signals, f"{path}.case[{ci}].stmt[{j}]", errors)
    elif isinstance(stmt, Assert):
        _validate_expr(stmt.cond, known_signals, f"{path}.cond", errors)
    elif isinstance(stmt, Print):
        for j, a in enumerate(stmt.args):
            _validate_expr(a, known_signals, f"{path}.arg[{j}]", errors)


def _validate_expr(
    expr: Expr,
    known_signals: set[str],
    path: str,
    errors: list[ValidationError],
) -> None:
    """Validate an expression tree — check signal references exist."""
    if isinstance(expr, Const):
        pass
    elif isinstance(expr, SignalRef):
        if expr.name not in known_signals:
            _add_error(errors, path, f"reference to unknown signal: {expr.name}")
    elif isinstance(expr, Unary):
        _validate_expr(expr.operand, known_signals, f"{path}.operand", errors)
    elif isinstance(expr, Binary):
        _validate_expr(expr.left, known_signals, f"{path}.left", errors)
        _validate_expr(expr.right, known_signals, f"{path}.right", errors)
    elif isinstance(expr, Mux):
        _validate_expr(expr.sel, known_signals, f"{path}.sel", errors)
        _validate_expr(expr.if_true, known_signals, f"{path}.if_true", errors)
        _validate_expr(expr.if_false, known_signals, f"{path}.if_false", errors)
    elif isinstance(expr, Concat):
        for j, p in enumerate(expr.parts):
            _validate_expr(p, known_signals, f"{path}.part[{j}]", errors)
    elif isinstance(expr, Slice):
        _validate_expr(expr.value, known_signals, f"{path}.value", errors)
