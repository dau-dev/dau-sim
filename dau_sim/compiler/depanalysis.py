"""Combinational dependency analysis.

Given a set of combinational assignments, determine:
1. Which signals each assignment reads (dependencies)
2. Topological evaluation order
3. Whether combinational loops exist
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from dau_sim.ir.expr import Binary, Concat, Const, Expr, Mux, SignalRef, Slice, Unary
from dau_sim.ir.stmt import Assign, IfElse, Stmt, Switch


def collect_reads(expr: Expr) -> set[str]:
    """Collect all signal names read by an expression."""
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, SignalRef):
        return {expr.name}
    if isinstance(expr, Unary):
        return collect_reads(expr.operand)
    if isinstance(expr, Binary):
        return collect_reads(expr.left) | collect_reads(expr.right)
    if isinstance(expr, Mux):
        return collect_reads(expr.sel) | collect_reads(expr.if_true) | collect_reads(expr.if_false)
    if isinstance(expr, Concat):
        result: set[str] = set()
        for p in expr.parts:
            result |= collect_reads(p)
        return result
    if isinstance(expr, Slice):
        return collect_reads(expr.value)
    return set()


def collect_stmt_reads(stmt: Stmt) -> set[str]:
    """Collect all signal names read by a statement (recursing into sub-stmts)."""
    if isinstance(stmt, Assign):
        return collect_reads(stmt.value)
    if isinstance(stmt, IfElse):
        reads = collect_reads(stmt.cond)
        for s in stmt.then_body:
            reads |= collect_stmt_reads(s)
        for s in stmt.else_body:
            reads |= collect_stmt_reads(s)
        return reads
    if isinstance(stmt, Switch):
        reads = collect_reads(stmt.test)
        for _, stmts in stmt.cases:
            for s in stmts:
                reads |= collect_stmt_reads(s)
        return reads
    return set()


def collect_stmt_writes(stmt: Stmt) -> set[str]:
    """Collect all signal names written by a statement."""
    if isinstance(stmt, Assign):
        return {stmt.target}
    if isinstance(stmt, IfElse):
        writes: set[str] = set()
        for s in stmt.then_body:
            writes |= collect_stmt_writes(s)
        for s in stmt.else_body:
            writes |= collect_stmt_writes(s)
        return writes
    if isinstance(stmt, Switch):
        writes = set()
        for _, stmts in stmt.cases:
            for s in stmts:
                writes |= collect_stmt_writes(s)
        return writes
    return set()


@dataclass
class Assignment:
    """A single combinational assignment with its dependency info.

    ``index`` refers to the position of the original statement or comb block.
    """

    index: int
    writes: set[str]
    reads: set[str]
    stmts: tuple[Stmt, ...]

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Assignment):
            return NotImplemented
        return self.index == other.index


@dataclass(frozen=True)
class AssignmentComponent:
    """A connected set of combinational assignments.

    Components are disconnected from each other in the assignment dependency
    graph, so they can be analyzed or scheduled independently.
    """

    assignments: tuple[Assignment, ...]
    reads: frozenset[str]
    writes: frozenset[str]
    signals: frozenset[str]


@dataclass
class CombLoopError(Exception):
    """Raised when combinational assignments form a cycle."""

    cycle: list[str]  # signal names forming the loop

    def __str__(self) -> str:
        path = " → ".join(self.cycle)
        return f"Combinational loop detected: {path}"


def build_assignments(stmts_list: list[tuple[int, tuple[Stmt, ...]]]) -> list[Assignment]:
    """Build Assignment objects from indexed statement groups.

    Each entry is (index, stmts) where stmts is a tuple of statements
    from a single CombBlock.
    """
    assignments = []
    for idx, stmts in stmts_list:
        writes: set[str] = set()
        reads: set[str] = set()
        for s in stmts:
            writes |= collect_stmt_writes(s)
            reads |= collect_stmt_reads(s)
        assignments.append(Assignment(index=idx, writes=writes, reads=reads, stmts=stmts))
    return assignments


def topological_sort(assignments: list[Assignment]) -> list[Assignment]:
    """Sort assignments in dependency order (consumers after producers).

    Raises CombLoopError if a cycle is detected.
    Returns the sorted list.
    """
    if not assignments:
        return []

    # Build a graph: for each assignment, find which other assignments
    # must execute before it (i.e., assignments whose writes it reads).
    # Map signal→producing assignment index
    sig_to_producer: dict[str, list[int]] = {}
    for i, a in enumerate(assignments):
        for w in a.writes:
            sig_to_producer.setdefault(w, []).append(i)

    n = len(assignments)
    adj: list[list[int]] = [[] for _ in range(n)]
    in_degree = [0] * n

    for i, a in enumerate(assignments):
        deps: set[int] = set()
        for r in a.reads:
            for p in sig_to_producer.get(r, []):
                if p != i:
                    deps.add(p)
        for d in deps:
            adj[d].append(i)
            in_degree[i] += 1

    # Kahn's algorithm
    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != n:
        # Find the cycle for reporting
        cycle = _find_cycle(assignments, sig_to_producer)
        raise CombLoopError(cycle=cycle)

    return [assignments[i] for i in order]


def partition_assignments(assignments: list[Assignment]) -> list[AssignmentComponent]:
    """Partition assignments into disconnected dependency components.

    The input is typically already topologically sorted. Each returned
    component preserves that ordering within the component.
    """
    if not assignments:
        return []

    producer_for_signal: dict[str, list[int]] = {}
    for idx, assignment in enumerate(assignments):
        for written in assignment.writes:
            producer_for_signal.setdefault(written, []).append(idx)

    adjacency: list[set[int]] = [set() for _ in range(len(assignments))]
    for idx, assignment in enumerate(assignments):
        deps: set[int] = set()
        for read in assignment.reads:
            for prod in producer_for_signal.get(read, []):
                if prod != idx:
                    deps.add(prod)
        for written in assignment.writes:
            for peer in producer_for_signal.get(written, []):
                if peer != idx:
                    deps.add(peer)
        for dep in deps:
            adjacency[idx].add(dep)
            adjacency[dep].add(idx)

    visited = [False] * len(assignments)
    components: list[AssignmentComponent] = []
    for start in range(len(assignments)):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component_indices: set[int] = set()
        while stack:
            cur = stack.pop()
            component_indices.add(cur)
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)

        component_assignments = tuple(assignment for idx, assignment in enumerate(assignments) if idx in component_indices)
        reads = frozenset().union(*(assignment.reads for assignment in component_assignments))
        writes = frozenset().union(*(assignment.writes for assignment in component_assignments))
        components.append(
            AssignmentComponent(
                assignments=component_assignments,
                reads=reads,
                writes=writes,
                signals=reads | writes,
            )
        )

    return components


def build_component_signal_index(components: list[AssignmentComponent] | tuple[AssignmentComponent, ...]) -> dict[str, tuple[int, ...]]:
    """Map each signal to the component ids that depend on it."""
    signal_to_components: dict[str, list[int]] = {}
    for component_id, component in enumerate(components):
        for signal in component.signals:
            signal_to_components.setdefault(signal, []).append(component_id)
    return {signal: tuple(component_ids) for signal, component_ids in signal_to_components.items()}


def affected_component_ids(
    signal_to_components: dict[str, tuple[int, ...]],
    changed_signals: set[str],
) -> tuple[int, ...]:
    """Return sorted component ids affected by the changed signals."""
    if not changed_signals:
        return ()

    affected: set[int] = set()
    for signal in changed_signals:
        affected.update(signal_to_components.get(signal, ()))
    return tuple(sorted(affected))


def _find_cycle(
    assignments: list[Assignment],
    sig_to_producer: dict[str, list[int]],
) -> list[str]:
    """Find a signal-level cycle for error reporting."""
    # Build signal-level graph: signal → signals it depends on
    sig_deps: dict[str, set[str]] = {}
    for a in assignments:
        for w in a.writes:
            sig_deps.setdefault(w, set())
            for r in a.reads:
                if r != w:
                    sig_deps[w].add(r)

    # DFS to find cycle
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {s: WHITE for s in sig_deps}
    parent: dict[str, str | None] = {s: None for s in sig_deps}

    def dfs(u: str) -> list[str] | None:
        color[u] = GRAY
        for v in sig_deps.get(u, set()):
            if v not in color:
                continue
            if color[v] == GRAY:
                # Found cycle, reconstruct
                cycle = [v, u]
                cur = u
                while cur != v:
                    cur = parent.get(cur)
                    if cur is None:
                        break
                    cycle.append(cur)
                cycle.reverse()
                return cycle
            if color[v] == WHITE:
                parent[v] = u
                result = dfs(v)
                if result is not None:
                    return result
        color[u] = BLACK
        return None

    for s in sig_deps:
        if color[s] == WHITE:
            result = dfs(s)
            if result is not None:
                return result

    return ["<unknown>"]
