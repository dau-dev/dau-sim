from ccflow import BaseModel
from pydantic import ConfigDict

from dau_sim.ir.expr import Expr


class Stmt(BaseModel):
    """Base class for all IR statements."""

    model_config = ConfigDict(frozen=True)


class Assign(Stmt):
    """Assign expression value to a signal.

    target is the signal name (str), value is the expression to evaluate.
    """

    target: str
    value: Expr


class IfElse(Stmt):
    """Conditional statement.

    if cond: then_stmts else: else_stmts
    """

    cond: Expr
    then_body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...] = ()


class Switch(Stmt):
    """Multi-way branch (case/casez/casex).

    test is the expression to match. cases is a tuple of (pattern, stmts) pairs.
    Pattern is an int (matched by equality) or None for the default case.
    """

    test: Expr
    cases: tuple[tuple[int | None, tuple[Stmt, ...]], ...]


class Assert(Stmt):
    """Assertion statement.

    Checks condition at simulation time, reports message on failure.
    """

    cond: Expr
    message: str = ""


class Print(Stmt):
    """Print statement ($display equivalent).

    format_str uses Python-style {} placeholders. args are expressions.
    """

    format_str: str
    args: tuple[Expr, ...] = ()


class Delay(Stmt):
    """Delay statement (#N equivalent).

    Pauses execution for *ticks* simulation time units.
    Only meaningful inside initial blocks or testbench procedures.
    """

    ticks: int


class Finish(Stmt):
    """Halt simulation ($finish equivalent).

    When executed, raises SimulationFinish to stop the simulation cleanly.
    """

    exit_code: int = 0


class ReadMem(Stmt):
    """Load memory contents from a file (``$readmemh``/``$readmemb``).

    *path* is the file path (resolved relative to the caller).
    *mem_name* is the name of the IR ``Memory`` to populate.
    *is_hex* selects hex (True, ``$readmemh``) or binary (False, ``$readmemb``).
    Optional *start_addr* / *end_addr* limit the address range filled.
    """

    path: str
    mem_name: str
    is_hex: bool = True
    start_addr: int | None = None
    end_addr: int | None = None
