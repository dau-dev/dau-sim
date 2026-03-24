from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class ValidationError:
    """A single validation error."""

    path: str  # e.g. "module.comb_block[0].stmt[1]"
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Collected validation errors."""

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add(self, path: str, message: str) -> None:
        self.errors.append(ValidationError(path, message))

    def __str__(self) -> str:
        if self.ok:
            return "OK"
        return "\n".join(str(e) for e in self.errors)


def validate_module(module: Module) -> ValidationResult:
    """Validate a module for well-formedness."""
    result = ValidationResult()
    prefix = f"module({module.name})"

    # Collect all known signal names
    known_signals = module.all_signal_names

    # Check for duplicate signal names
    seen: set[str] = set()
    for p in module.ports:
        if p.name in seen:
            result.add(prefix, f"duplicate signal name: {p.name}")
        seen.add(p.name)
    for s in module.signals:
        if s.name in seen:
            result.add(prefix, f"duplicate signal name: {s.name}")
        seen.add(s.name)

    # Check clock domains reference valid signals
    for cd in module.clock_domains:
        if cd.clk not in known_signals:
            result.add(
                f"{prefix}.clock_domain({cd.name})",
                f"clock signal '{cd.clk}' not found",
            )
        if cd.rst and cd.rst not in known_signals:
            result.add(
                f"{prefix}.clock_domain({cd.name})",
                f"reset signal '{cd.rst}' not found",
            )

    # Check sequentially blocks reference valid clock domains
    domain_names = {cd.name for cd in module.clock_domains}
    for i, sb in enumerate(module.seq_blocks):
        if sb.domain not in domain_names:
            result.add(
                f"{prefix}.seq_block[{i}]",
                f"unknown clock domain: {sb.domain}",
            )

    # Validate all blocks' statements
    for i, cb in enumerate(module.comb_blocks):
        _validate_block_stmts(cb.stmts, known_signals, f"{prefix}.comb_block[{i}]", result)
    for i, sb in enumerate(module.seq_blocks):
        _validate_block_stmts(sb.stmts, known_signals, f"{prefix}.seq_block[{i}]", result)
    for i, ib in enumerate(module.init_blocks):
        _validate_block_stmts(ib.stmts, known_signals, f"{prefix}.init_block[{i}]", result)

    return result


def _validate_block_stmts(
    stmts: tuple[Stmt, ...],
    known_signals: set[str],
    path: str,
    result: ValidationResult,
) -> None:
    """Validate statements within a block."""
    for i, stmt in enumerate(stmts):
        _validate_stmt(stmt, known_signals, f"{path}.stmt[{i}]", result)


def _validate_stmt(
    stmt: Stmt,
    known_signals: set[str],
    path: str,
    result: ValidationResult,
) -> None:
    """Validate a single statement."""
    if isinstance(stmt, Assign):
        if stmt.target not in known_signals:
            result.add(path, f"assignment to unknown signal: {stmt.target}")
        _validate_expr(stmt.value, known_signals, f"{path}.value", result)
    elif isinstance(stmt, IfElse):
        _validate_expr(stmt.cond, known_signals, f"{path}.cond", result)
        for j, s in enumerate(stmt.then_body):
            _validate_stmt(s, known_signals, f"{path}.then[{j}]", result)
        for j, s in enumerate(stmt.else_body):
            _validate_stmt(s, known_signals, f"{path}.else[{j}]", result)
    elif isinstance(stmt, Switch):
        _validate_expr(stmt.test, known_signals, f"{path}.test", result)
        for ci, (_, stmts) in enumerate(stmt.cases):
            for j, s in enumerate(stmts):
                _validate_stmt(s, known_signals, f"{path}.case[{ci}].stmt[{j}]", result)
    elif isinstance(stmt, Assert):
        _validate_expr(stmt.cond, known_signals, f"{path}.cond", result)
    elif isinstance(stmt, Print):
        for j, a in enumerate(stmt.args):
            _validate_expr(a, known_signals, f"{path}.arg[{j}]", result)


def _validate_expr(
    expr: Expr,
    known_signals: set[str],
    path: str,
    result: ValidationResult,
) -> None:
    """Validate an expression tree — check signal references exist."""
    if isinstance(expr, Const):
        pass
    elif isinstance(expr, SignalRef):
        if expr.name not in known_signals:
            result.add(path, f"reference to unknown signal: {expr.name}")
    elif isinstance(expr, Unary):
        _validate_expr(expr.operand, known_signals, f"{path}.operand", result)
    elif isinstance(expr, Binary):
        _validate_expr(expr.left, known_signals, f"{path}.left", result)
        _validate_expr(expr.right, known_signals, f"{path}.right", result)
    elif isinstance(expr, Mux):
        _validate_expr(expr.sel, known_signals, f"{path}.sel", result)
        _validate_expr(expr.if_true, known_signals, f"{path}.if_true", result)
        _validate_expr(expr.if_false, known_signals, f"{path}.if_false", result)
    elif isinstance(expr, Concat):
        for j, p in enumerate(expr.parts):
            _validate_expr(p, known_signals, f"{path}.part[{j}]", result)
    elif isinstance(expr, Slice):
        _validate_expr(expr.value, known_signals, f"{path}.value", result)
