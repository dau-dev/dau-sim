"""IR-to-Python code generator.

Compiles IR statement/expression trees into flat Python functions,
eliminating tree-walking overhead in the simulation hot path.

Generated functions operate on a flat signal array (list[int]) indexed
by integer signal IDs, with shapes/masks precomputed at compile time.
"""

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
    SysRandom,
    Unary,
    UnaryOp,
)
from dau_sim.ir.stmt import Assert, Assign, Delay, Finish, IfElse, Print, ReadMem, Stmt, Switch
from dau_sim.ir.types import Shape


def _width_mask(width: int) -> int:
    return (1 << width) - 1 if width > 0 else 0


class CodeGen:
    """Compiles IR trees into Python source that operates on a signal array.

    Usage::

        cg = CodeGen(shapes)
        fn = cg.compile_block(stmts)
        # fn(signals, changed_signals) — mutates signals in-place
    """

    def __init__(self, shapes: dict[str, Shape]):
        self._shapes = shapes
        # Map signal names to integer indices for array-based storage
        self._sig_names = sorted(shapes.keys())
        self._sig_index: dict[str, int] = {n: i for i, n in enumerate(self._sig_names)}
        # Precompute masks
        self._masks: dict[str, int] = {n: _width_mask(s.width) for n, s in shapes.items()}
        self._counter = 0

    @property
    def signal_names(self) -> list[str]:
        return self._sig_names

    @property
    def signal_index(self) -> dict[str, int]:
        return self._sig_index

    def _tmp(self) -> str:
        self._counter += 1
        return f"_t{self._counter}"

    def compile_block(
        self,
        stmts: tuple[Stmt, ...],
        *,
        name: str = "_compiled_block",
    ) -> callable:
        """Compile a statement block into a callable ``fn(S, changed)``."""
        self._counter = 0
        lines: list[str] = []
        lines.append(f"def {name}(S, changed):")
        body = self._compile_stmts(stmts, indent=1)
        if not body:
            lines.append("    pass")
        else:
            lines.extend(body)
        source = "\n".join(lines)
        globs = self._make_globals()
        code = compile(source, f"<codegen:{name}>", "exec")
        ns: dict = {}
        exec(code, globs, ns)  # noqa: S102
        fn = ns[name]
        fn._codegen_source = source
        return fn

    def compile_expr_block(
        self,
        stmts: tuple[Stmt, ...],
        read_signals: set[str],
        write_signals: set[str],
        *,
        name: str = "_compiled_block",
    ) -> callable:
        """Compile a block with local-variable optimization.

        Generates code that loads all read/written signals into locals
        at the top and stores changed signals back at the bottom.
        """
        self._counter = 0
        lines: list[str] = []
        lines.append(f"def {name}(S, changed):")

        indent = 1
        pad = "    " * indent

        # All signals that appear in local variables
        all_locals = read_signals | write_signals

        # Load all local signals from array
        for sig in sorted(all_locals):
            idx = self._sig_index[sig]
            local = self._local_name(sig)
            lines.append(f"{pad}{local} = S[{idx}]")

        # Compile body using locals for all reads and writes
        body = self._compile_stmts_local(stmts, indent=indent, local_reads=all_locals)
        if body:
            lines.extend(body)

        # Store writes back
        for sig in sorted(write_signals):
            idx = self._sig_index[sig]
            local = self._local_name(sig)
            lines.append(f"{pad}if S[{idx}] != {local}:")
            lines.append(f"{pad}    S[{idx}] = {local}")
            lines.append(f"{pad}    changed.add({idx!r})")

        source = "\n".join(lines)
        globs = self._make_globals()
        code = compile(source, f"<codegen:{name}>", "exec")
        ns: dict = {}
        exec(code, globs, ns)  # noqa: S102
        fn = ns[name]
        fn._codegen_source = source
        return fn

    def _local_name(self, sig: str) -> str:
        """Generate a valid Python local variable name for a signal."""
        return f"_s_{sig.replace('.', '_').replace('[', '_').replace(']', '_').replace('$', '_')}"

    def _make_globals(self) -> dict:
        """Build the globals dict for compiled code."""
        from dau_sim.compiler.compile import SimulationFinish
        from dau_sim.compiler.eval import _sys_random_rng

        return {
            "__builtins__": {"int": int, "print": print, "abs": abs},
            "_SimulationFinish": SimulationFinish,
            "_sys_random_rng": _sys_random_rng,
        }

    # ── Statement compilation ──────────────────────────────────────

    def _compile_stmts(self, stmts: tuple[Stmt, ...], indent: int) -> list[str]:
        lines: list[str] = []
        for stmt in stmts:
            lines.extend(self._compile_stmt(stmt, indent))
        return lines

    def _compile_stmt(self, stmt: Stmt, indent: int) -> list[str]:
        pad = "    " * indent
        st = type(stmt)

        if st is Assign:
            return self._compile_assign(stmt, indent)

        if st is IfElse:
            lines: list[str] = []
            cond_expr = self._emit_expr(stmt.cond)
            lines.append(f"{pad}if {cond_expr}:")
            then = self._compile_stmts(stmt.then_body, indent + 1)
            lines.extend(then if then else [f"{pad}    pass"])
            if stmt.else_body:
                lines.append(f"{pad}else:")
                else_ = self._compile_stmts(stmt.else_body, indent + 1)
                lines.extend(else_ if else_ else [f"{pad}    pass"])
            return lines

        if st is Switch:
            return self._compile_switch(stmt, indent)

        if st is Assert:
            cond_expr = self._emit_expr(stmt.cond)
            msg = repr(stmt.message or "assertion failed")
            return [f"{pad}if not ({cond_expr}): raise AssertionError({msg})"]

        if st is Print:
            args = ", ".join(self._emit_expr(a) for a in stmt.args)
            return [f"{pad}print({stmt.format_str!r}.format({args}))"]

        if st is Finish:
            return [f"{pad}raise _SimulationFinish({stmt.exit_code})"]

        if st is Delay:
            return []  # no-op in synchronous execution

        if st is ReadMem:
            return []  # handled at init-block level

        raise TypeError(f"Unknown statement type: {st.__name__}")

    def _compile_assign(self, stmt: Assign, indent: int) -> list[str]:
        pad = "    " * indent
        idx = self._sig_index[stmt.target]
        shape = self._shapes[stmt.target]
        val_expr = self._emit_expr(stmt.value)
        masked = self._emit_mask(val_expr, shape)
        return [
            f"{pad}_v = {masked}",
            f"{pad}if S[{idx}] != _v:",
            f"{pad}    changed.add({idx!r})",
            f"{pad}    S[{idx}] = _v",
        ]

    def _compile_switch(self, stmt: Switch, indent: int) -> list[str]:
        pad = "    " * indent
        lines: list[str] = []
        test_expr = self._emit_expr(stmt.test)
        tmp = self._tmp()
        lines.append(f"{pad}{tmp} = {test_expr}")

        first = True
        default_stmts = None
        for pattern, body in stmt.cases:
            if pattern is None:
                default_stmts = body
                continue
            kw = "if" if first else "elif"
            lines.append(f"{pad}{kw} {tmp} == {pattern!r}:")
            inner = self._compile_stmts(body, indent + 1)
            lines.extend(inner if inner else [f"{pad}    pass"])
            first = False

        if default_stmts is not None:
            if first:
                # Only default case
                lines.extend(self._compile_stmts(default_stmts, indent))
            else:
                lines.append(f"{pad}else:")
                inner = self._compile_stmts(default_stmts, indent + 1)
                lines.extend(inner if inner else [f"{pad}    pass"])

        return lines

    # ── Expression emission ────────────────────────────────────────

    def _emit_expr(self, expr: Expr) -> str:
        """Emit a Python expression string for an IR expression."""
        et = type(expr)

        if et is Const:
            return self._emit_const(expr)

        if et is SignalRef:
            return self._emit_signal_ref(expr)

        if et is Unary:
            return self._emit_unary(expr)

        if et is Binary:
            return self._emit_binary(expr)

        if et is Mux:
            sel = self._emit_expr(expr.sel)
            t = self._emit_expr(expr.if_true)
            f = self._emit_expr(expr.if_false)
            return self._emit_mask(f"({t} if {sel} else {f})", expr.shape)

        if et is Concat:
            return self._emit_concat(expr)

        if et is Slice:
            return self._emit_slice(expr)

        if et is SysRandom:
            return self._emit_sysrandom(expr)

        raise TypeError(f"Unknown expression type: {et.__name__}")

    def _emit_const(self, expr: Const) -> str:
        width = expr.shape.width
        if width == 0:
            return "0"
        mask = _width_mask(width)
        raw = expr.value & mask
        if expr.shape.signed and ((raw >> (width - 1)) & 1):
            raw -= 1 << width
        return repr(raw)

    def _emit_signal_ref(self, expr: SignalRef) -> str:
        idx = self._sig_index[expr.name]
        width = expr.shape.width
        if width == 0:
            return "0"
        mask = _width_mask(width)
        base = f"S[{idx}]"
        if expr.shape.signed:
            return self._emit_mask(base, expr.shape)
        # Unsigned: just mask
        return f"({base} & {mask})"

    def _emit_unary(self, expr: Unary) -> str:
        a = self._emit_expr(expr.operand)
        op = expr.op

        if op is UnaryOp.NOT:
            return self._emit_mask(f"(~{a})", expr.shape)

        if op is UnaryOp.NEG:
            return self._emit_mask(f"(-{a})", expr.shape)

        if op is UnaryOp.BOOL:
            return f"(1 if {a} != 0 else 0)"

        if op is UnaryOp.RED_AND:
            all_ones = _width_mask(expr.operand.shape.width)
            return f"(1 if ({a} & {all_ones}) == {all_ones} else 0)"

        if op is UnaryOp.RED_OR:
            return f"(1 if {a} != 0 else 0)"

        if op is UnaryOp.RED_XOR:
            mask = _width_mask(expr.operand.shape.width)
            return f"(({a} & {mask}).bit_count() & 1)"

        raise ValueError(f"Unknown unary op: {op}")

    def _emit_binary(self, expr: Binary) -> str:
        left = self._emit_expr(expr.left)
        right = self._emit_expr(expr.right)
        op = expr.op
        out = expr.shape

        if op is BinaryOp.ADD:
            return self._emit_mask(f"({left} + {right})", out)
        if op is BinaryOp.SUB:
            return self._emit_mask(f"({left} - {right})", out)
        if op is BinaryOp.MUL:
            return self._emit_mask(f"({left} * {right})", out)
        if op is BinaryOp.DIV:
            if expr.left.shape.signed or expr.right.shape.signed:
                return self._emit_mask(f"(int({left} / {right}) if {right} != 0 else 0)", out)
            return self._emit_mask(f"({left} // {right} if {right} != 0 else 0)", out)
        if op is BinaryOp.MOD:
            return self._emit_mask(f"({left} % {right} if {right} != 0 else 0)", out)
        if op is BinaryOp.AND:
            return self._emit_mask(f"({left} & {right})", out)
        if op is BinaryOp.OR:
            return self._emit_mask(f"({left} | {right})", out)
        if op is BinaryOp.XOR:
            return self._emit_mask(f"({left} ^ {right})", out)
        if op is BinaryOp.SHL:
            return self._emit_mask(f"({left} << {right})", out)
        if op is BinaryOp.SHR:
            if expr.left.shape.signed:
                return self._emit_mask(f"({left} >> {right})", out)
            # Unsigned: mask left first
            lmask = _width_mask(expr.left.shape.width)
            return self._emit_mask(f"(({left} & {lmask}) >> {right})", out)

        # Comparison operators → 1-bit result
        _cmp_ops = {
            BinaryOp.EQ: "==",
            BinaryOp.NE: "!=",
            BinaryOp.LT: "<",
            BinaryOp.LE: "<=",
            BinaryOp.GT: ">",
            BinaryOp.GE: ">=",
        }
        if op in _cmp_ops:
            return f"(1 if {left} {_cmp_ops[op]} {right} else 0)"

        if op is BinaryOp.LOGIC_AND:
            return f"(1 if ({left} != 0 and {right} != 0) else 0)"
        if op is BinaryOp.LOGIC_OR:
            return f"(1 if ({left} != 0 or {right} != 0) else 0)"

        raise ValueError(f"Unknown binary op: {op}")

    def _emit_concat(self, expr: Concat) -> str:
        if not expr.parts:
            return "0"
        # Build up MSB-first: result = (part0 << (w1+w2+...)) | (part1 << (w2+...)) | ...
        parts_code: list[str] = []
        shift = sum(p.shape.width for p in expr.parts)
        for part in expr.parts:
            shift -= part.shape.width
            p = self._emit_expr(part)
            pmask = _width_mask(part.shape.width)
            if shift > 0:
                parts_code.append(f"(({p} & {pmask}) << {shift})")
            else:
                parts_code.append(f"({p} & {pmask})")
        combined = " | ".join(parts_code)
        return self._emit_mask(f"({combined})", expr.shape)

    def _emit_slice(self, expr: Slice) -> str:
        val = self._emit_expr(expr.value)
        low = expr.low
        width = expr.high - expr.low
        mask = _width_mask(width)
        if low == 0:
            extracted = f"({val} & {mask})"
        else:
            extracted = f"(({val} >> {low}) & {mask})"
        return self._emit_mask(extracted, expr.shape)

    def _emit_sysrandom(self, expr: SysRandom) -> str:
        if expr.seed is not None:
            seed = self._emit_expr(expr.seed)
            return self._emit_mask(
                f"(_sys_random_rng.seed({seed}) or _sys_random_rng.randint(-(1 << 31), (1 << 31) - 1))",
                expr.shape,
            )
        return self._emit_mask(
            "_sys_random_rng.randint(-(1 << 31), (1 << 31) - 1)",
            expr.shape,
        )

    def _emit_mask(self, expr_str: str, shape: Shape) -> str:
        """Wrap an expression with mask/sign-extension."""
        width = shape.width
        if width == 0:
            return "0"
        mask = _width_mask(width)
        if shape.signed:
            sign_bit = 1 << (width - 1)
            # Two's complement sign extension: ((v & mask) ^ sign) - sign
            return f"(({expr_str} & {mask}) ^ {sign_bit}) - {sign_bit}"
        return f"({expr_str} & {mask})"

    # ── Local-variable optimized compilation ───────────────────────

    def _compile_stmts_local(
        self,
        stmts: tuple[Stmt, ...],
        indent: int,
        local_reads: set[str],
    ) -> list[str]:
        lines: list[str] = []
        for stmt in stmts:
            lines.extend(self._compile_stmt_local(stmt, indent, local_reads))
        return lines

    def _compile_stmt_local(
        self,
        stmt: Stmt,
        indent: int,
        local_reads: set[str],
    ) -> list[str]:
        pad = "    " * indent
        st = type(stmt)

        if st is Assign:
            return self._compile_assign_local(stmt, indent, local_reads)

        if st is IfElse:
            lines: list[str] = []
            cond_expr = self._emit_expr_local(stmt.cond, local_reads)
            lines.append(f"{pad}if {cond_expr}:")
            then = self._compile_stmts_local(stmt.then_body, indent + 1, local_reads)
            lines.extend(then if then else [f"{pad}    pass"])
            if stmt.else_body:
                lines.append(f"{pad}else:")
                else_ = self._compile_stmts_local(stmt.else_body, indent + 1, local_reads)
                lines.extend(else_ if else_ else [f"{pad}    pass"])
            return lines

        if st is Switch:
            return self._compile_switch_local(stmt, indent, local_reads)

        if st is Assert:
            cond_expr = self._emit_expr_local(stmt.cond, local_reads)
            msg = repr(stmt.message or "assertion failed")
            return [f"{pad}if not ({cond_expr}): raise AssertionError({msg})"]

        if st is Print:
            args = ", ".join(self._emit_expr_local(a, local_reads) for a in stmt.args)
            return [f"{pad}print({stmt.format_str!r}.format({args}))"]

        if st is Finish:
            return [f"{pad}raise _SimulationFinish({stmt.exit_code})"]

        if st is Delay:
            return []

        if st is ReadMem:
            return []

        raise TypeError(f"Unknown statement type: {st.__name__}")

    def _compile_assign_local(
        self,
        stmt: Assign,
        indent: int,
        local_reads: set[str],
    ) -> list[str]:
        pad = "    " * indent
        shape = self._shapes[stmt.target]
        val_expr = self._emit_expr_local(stmt.value, local_reads)
        masked = self._emit_mask(val_expr, shape)

        if stmt.target in local_reads:
            local = self._local_name(stmt.target)
            return [f"{pad}{local} = {masked}"]
        else:
            # Write-only signal (not in local_reads set) — write directly to array
            idx = self._sig_index[stmt.target]
            return [
                f"{pad}_v = {masked}",
                f"{pad}if S[{idx}] != _v:",
                f"{pad}    changed.add({idx!r})",
                f"{pad}    S[{idx}] = _v",
            ]

    def _compile_switch_local(
        self,
        stmt: Switch,
        indent: int,
        local_reads: set[str],
    ) -> list[str]:
        pad = "    " * indent
        lines: list[str] = []
        test_expr = self._emit_expr_local(stmt.test, local_reads)
        tmp = self._tmp()
        lines.append(f"{pad}{tmp} = {test_expr}")

        first = True
        default_stmts = None
        for pattern, body in stmt.cases:
            if pattern is None:
                default_stmts = body
                continue
            kw = "if" if first else "elif"
            lines.append(f"{pad}{kw} {tmp} == {pattern!r}:")
            inner = self._compile_stmts_local(body, indent + 1, local_reads)
            lines.extend(inner if inner else [f"{pad}    pass"])
            first = False

        if default_stmts is not None:
            if first:
                lines.extend(self._compile_stmts_local(default_stmts, indent, local_reads))
            else:
                lines.append(f"{pad}else:")
                inner = self._compile_stmts_local(default_stmts, indent + 1, local_reads)
                lines.extend(inner if inner else [f"{pad}    pass"])

        return lines

    def _emit_expr_local(self, expr: Expr, local_reads: set[str]) -> str:
        """Emit expression using local variables for signal reads where possible."""
        et = type(expr)

        if et is Const:
            return self._emit_const(expr)

        if et is SignalRef:
            if expr.name in local_reads:
                local = self._local_name(expr.name)
                width = expr.shape.width
                if width == 0:
                    return "0"
                mask = _width_mask(width)
                if expr.shape.signed:
                    return self._emit_mask(local, expr.shape)
                return f"({local} & {mask})"
            return self._emit_signal_ref(expr)

        if et is Unary:
            a = self._emit_expr_local(expr.operand, local_reads)
            return self._emit_unary_raw(expr.op, a, expr.operand.shape, expr.shape)

        if et is Binary:
            left = self._emit_expr_local(expr.left, local_reads)
            right = self._emit_expr_local(expr.right, local_reads)
            return self._emit_binary_raw(expr.op, left, right, expr.left.shape, expr.right.shape, expr.shape)

        if et is Mux:
            sel = self._emit_expr_local(expr.sel, local_reads)
            t = self._emit_expr_local(expr.if_true, local_reads)
            f = self._emit_expr_local(expr.if_false, local_reads)
            return self._emit_mask(f"({t} if {sel} else {f})", expr.shape)

        if et is Concat:
            return self._emit_concat_local(expr, local_reads)

        if et is Slice:
            return self._emit_slice_local(expr, local_reads)

        if et is SysRandom:
            return self._emit_sysrandom(expr)

        raise TypeError(f"Unknown expression type: {et.__name__}")

    def _emit_unary_raw(self, op: UnaryOp, a: str, a_shape: Shape, out_shape: Shape) -> str:
        if op is UnaryOp.NOT:
            return self._emit_mask(f"(~{a})", out_shape)
        if op is UnaryOp.NEG:
            return self._emit_mask(f"(-{a})", out_shape)
        if op is UnaryOp.BOOL:
            return f"(1 if {a} != 0 else 0)"
        if op is UnaryOp.RED_AND:
            all_ones = _width_mask(a_shape.width)
            return f"(1 if ({a} & {all_ones}) == {all_ones} else 0)"
        if op is UnaryOp.RED_OR:
            return f"(1 if {a} != 0 else 0)"
        if op is UnaryOp.RED_XOR:
            mask = _width_mask(a_shape.width)
            return f"(({a} & {mask}).bit_count() & 1)"
        raise ValueError(f"Unknown unary op: {op}")

    def _emit_binary_raw(
        self,
        op: BinaryOp,
        left: str,
        right: str,
        l_shape: Shape,
        r_shape: Shape,
        out_shape: Shape,
    ) -> str:
        if op is BinaryOp.ADD:
            return self._emit_mask(f"({left} + {right})", out_shape)
        if op is BinaryOp.SUB:
            return self._emit_mask(f"({left} - {right})", out_shape)
        if op is BinaryOp.MUL:
            return self._emit_mask(f"({left} * {right})", out_shape)
        if op is BinaryOp.DIV:
            if l_shape.signed or r_shape.signed:
                return self._emit_mask(f"(int({left} / {right}) if {right} != 0 else 0)", out_shape)
            return self._emit_mask(f"({left} // {right} if {right} != 0 else 0)", out_shape)
        if op is BinaryOp.MOD:
            return self._emit_mask(f"({left} % {right} if {right} != 0 else 0)", out_shape)
        if op is BinaryOp.AND:
            return self._emit_mask(f"({left} & {right})", out_shape)
        if op is BinaryOp.OR:
            return self._emit_mask(f"({left} | {right})", out_shape)
        if op is BinaryOp.XOR:
            return self._emit_mask(f"({left} ^ {right})", out_shape)
        if op is BinaryOp.SHL:
            return self._emit_mask(f"({left} << {right})", out_shape)
        if op is BinaryOp.SHR:
            if l_shape.signed:
                return self._emit_mask(f"({left} >> {right})", out_shape)
            lmask = _width_mask(l_shape.width)
            return self._emit_mask(f"(({left} & {lmask}) >> {right})", out_shape)
        _cmp_ops = {
            BinaryOp.EQ: "==",
            BinaryOp.NE: "!=",
            BinaryOp.LT: "<",
            BinaryOp.LE: "<=",
            BinaryOp.GT: ">",
            BinaryOp.GE: ">=",
        }
        if op in _cmp_ops:
            return f"(1 if {left} {_cmp_ops[op]} {right} else 0)"
        if op is BinaryOp.LOGIC_AND:
            return f"(1 if ({left} != 0 and {right} != 0) else 0)"
        if op is BinaryOp.LOGIC_OR:
            return f"(1 if ({left} != 0 or {right} != 0) else 0)"
        raise ValueError(f"Unknown binary op: {op}")

    def _emit_concat_local(self, expr: Concat, local_reads: set[str]) -> str:
        if not expr.parts:
            return "0"
        parts_code: list[str] = []
        shift = sum(p.shape.width for p in expr.parts)
        for part in expr.parts:
            shift -= part.shape.width
            p = self._emit_expr_local(part, local_reads)
            pmask = _width_mask(part.shape.width)
            if shift > 0:
                parts_code.append(f"(({p} & {pmask}) << {shift})")
            else:
                parts_code.append(f"({p} & {pmask})")
        combined = " | ".join(parts_code)
        return self._emit_mask(f"({combined})", expr.shape)

    def _emit_slice_local(self, expr: Slice, local_reads: set[str]) -> str:
        val = self._emit_expr_local(expr.value, local_reads)
        low = expr.low
        width = expr.high - expr.low
        mask = _width_mask(width)
        if low == 0:
            extracted = f"({val} & {mask})"
        else:
            extracted = f"(({val} >> {low}) & {mask})"
        return self._emit_mask(extracted, expr.shape)

    def build_fast_tick(
        self,
        domain_info: dict[str, dict],
    ) -> callable | None:
        """Generate a compiled tick function that inlines clock toggle + seq body.

        Returns None if any domain has a reset (fast path not applicable).
        The generated function signature is ``fn(S, clock_arr, tc)``
        and returns True if any domain fired, False otherwise.
        """
        from dau_sim.ir.types import EdgePolarity

        for dinfo in domain_info.values():
            if dinfo["rst_signal"] is not None:
                return None

        self._counter = 0
        lines: list[str] = ["def _fast_tick(S, clock_arr, tc):"]

        for d_i, (dname, dinfo) in enumerate(domain_info.items()):
            hpt = dinfo["half_period_ticks"]
            clk_signal = dinfo["clk_signal"]
            clk_idx = self._sig_index.get(clk_signal, -1)
            edge = dinfo["edge"]
            if edge == EdgePolarity.POSEDGE:
                fire_target = 1
            elif edge == EdgePolarity.NEGEDGE:
                fire_target = 0
            else:
                fire_target = -1

            # Determine base indent (skip modulo when hpt==1)
            if hpt == 1:
                base_indent = 1
            else:
                lines.append(f"    if tc % {hpt} == 0:")
                base_indent = 2

            pad = "    " * base_indent
            lines.append(f"{pad}old_clk_{d_i} = clock_arr[{d_i}]")
            lines.append(f"{pad}new_clk_{d_i} = 1 - old_clk_{d_i}")
            lines.append(f"{pad}clock_arr[{d_i}] = new_clk_{d_i}")
            if clk_idx >= 0:
                lines.append(f"{pad}S[{clk_idx}] = new_clk_{d_i}")

            # Collect seq stmts for this domain
            all_stmts: list = []
            for sb in dinfo.get("seq_blocks", []):
                all_stmts.extend(sb.stmts)
            stmts_tuple = tuple(all_stmts)

            if not stmts_tuple:
                continue

            reads, writes = collect_reads_writes(stmts_tuple)
            all_sigs = reads | writes

            # Determine seq indent based on edge check
            if fire_target == -1:
                # BOTH edges — always fires on toggle
                seq_indent = base_indent
            else:
                lines.append(f"{pad}if new_clk_{d_i} == {fire_target}:")
                seq_indent = base_indent + 1

            seq_pad = "    " * seq_indent

            # Load locals
            for sig in sorted(all_sigs):
                idx = self._sig_index[sig]
                local = self._local_name(sig)
                lines.append(f"{seq_pad}{local} = S[{idx}]")

            # Compile body
            body = self._compile_stmts_local(stmts_tuple, indent=seq_indent, local_reads=all_sigs)
            if body:
                lines.extend(body)

            # Store writes (no changed tracking)
            for sig in sorted(writes):
                idx = self._sig_index[sig]
                local = self._local_name(sig)
                lines.append(f"{seq_pad}if S[{idx}] != {local}:")
                lines.append(f"{seq_pad}    S[{idx}] = {local}")

        if len(lines) == 1:
            lines.append("    pass")

        source = "\n".join(lines)
        globs = self._make_globals()
        code = compile(source, "<codegen:_fast_tick>", "exec")
        ns: dict = {}
        exec(code, globs, ns)  # noqa: S102
        fn = ns["_fast_tick"]
        fn._codegen_source = source
        return fn


def collect_reads_writes(stmts: tuple[Stmt, ...]) -> tuple[set[str], set[str]]:
    """Collect all signal names read and written by a statement block."""
    reads: set[str] = set()
    writes: set[str] = set()
    for stmt in stmts:
        _collect_rw_stmt(stmt, reads, writes)
    return reads, writes


def _collect_rw_stmt(stmt: Stmt, reads: set[str], writes: set[str]) -> None:
    st = type(stmt)
    if st is Assign:
        writes.add(stmt.target)
        _collect_rw_expr(stmt.value, reads)
    elif st is IfElse:
        _collect_rw_expr(stmt.cond, reads)
        for s in stmt.then_body:
            _collect_rw_stmt(s, reads, writes)
        for s in stmt.else_body:
            _collect_rw_stmt(s, reads, writes)
    elif st is Switch:
        _collect_rw_expr(stmt.test, reads)
        for _, body in stmt.cases:
            for s in body:
                _collect_rw_stmt(s, reads, writes)
    elif st is Assert:
        _collect_rw_expr(stmt.cond, reads)
    elif st is Print:
        for a in stmt.args:
            _collect_rw_expr(a, reads)


def _collect_rw_expr(expr: Expr, reads: set[str]) -> None:
    et = type(expr)
    if et is SignalRef:
        reads.add(expr.name)
    elif et is Unary:
        _collect_rw_expr(expr.operand, reads)
    elif et is Binary:
        _collect_rw_expr(expr.left, reads)
        _collect_rw_expr(expr.right, reads)
    elif et is Mux:
        _collect_rw_expr(expr.sel, reads)
        _collect_rw_expr(expr.if_true, reads)
        _collect_rw_expr(expr.if_false, reads)
    elif et is Concat:
        for p in expr.parts:
            _collect_rw_expr(p, reads)
    elif et is Slice:
        _collect_rw_expr(expr.value, reads)
    elif et is SysRandom:
        if expr.seed is not None:
            _collect_rw_expr(expr.seed, reads)
