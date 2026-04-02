from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from time import perf_counter

from dau_sim.compiler import compile_module
from dau_sim.compiler.depanalysis import build_assignments
from dau_sim.ir.module import Module


@dataclass(frozen=True)
class BenchmarkResult:
    compile_seconds_median: float
    run_seconds_median: float
    cycles_per_second: float


@dataclass(frozen=True)
class NodeSeparationStats:
    comb_blocks: int
    dependency_edges: int
    connected_components: int
    largest_component: int
    singleton_components: int


@dataclass(frozen=True)
class PerformanceDelta:
    dau_cycles_per_second: float
    vs_amaranth_ratio: float | None
    vs_verilator_ratio: float | None
    multiplier_to_10x_current: float


def benchmark_module(
    module: Module,
    *,
    cycles: int,
    repeats: int,
    warmup: int = 1,
    inputs: dict[str, int] | None = None,
    clock_period: timedelta = timedelta(microseconds=1),
) -> BenchmarkResult:
    compile_times: list[float] = []
    run_times: list[float] = []

    for _ in range(max(1, repeats)):
        t0 = perf_counter()
        compiled = compile_module(module)
        t1 = perf_counter()
        compile_times.append(t1 - t0)

        for _ in range(max(0, warmup)):
            compiled.run(cycles=cycles, inputs=inputs, clock_period=clock_period)

        t2 = perf_counter()
        compiled.run(cycles=cycles, inputs=inputs, clock_period=clock_period)
        t3 = perf_counter()
        run_times.append(t3 - t2)

    run_med = median(run_times)
    return BenchmarkResult(
        compile_seconds_median=median(compile_times),
        run_seconds_median=run_med,
        cycles_per_second=(cycles / run_med) if run_med > 0 else float("inf"),
    )


def analyze_node_separation(module: Module) -> NodeSeparationStats:
    assignments = build_assignments([(i, cb.stmts) for i, cb in enumerate(module.comb_blocks)])
    n = len(assignments)

    producer_for_signal: dict[str, list[int]] = {}
    for idx, assignment in enumerate(assignments):
        for written in assignment.writes:
            producer_for_signal.setdefault(written, []).append(idx)

    adjacency: list[set[int]] = [set() for _ in range(n)]
    edge_count = 0
    for idx, assignment in enumerate(assignments):
        deps: set[int] = set()
        for read in assignment.reads:
            for prod in producer_for_signal.get(read, []):
                if prod != idx:
                    deps.add(prod)
        edge_count += len(deps)
        for dep in deps:
            adjacency[idx].add(dep)
            adjacency[dep].add(idx)

    visited = [False] * n
    component_sizes: list[int] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        component_sizes.append(size)

    if not component_sizes:
        component_sizes = [0]

    return NodeSeparationStats(
        comb_blocks=len(module.comb_blocks),
        dependency_edges=edge_count,
        connected_components=len(component_sizes),
        largest_component=max(component_sizes),
        singleton_components=sum(1 for sz in component_sizes if sz == 1),
    )


def evaluate_delta(
    dau_cycles_per_second: float,
    *,
    amaranth_cycles_per_second: float | None = None,
    verilator_cycles_per_second: float | None = None,
) -> PerformanceDelta:
    vs_amaranth = None
    if amaranth_cycles_per_second and amaranth_cycles_per_second > 0:
        vs_amaranth = dau_cycles_per_second / amaranth_cycles_per_second

    vs_verilator = None
    if verilator_cycles_per_second and verilator_cycles_per_second > 0:
        vs_verilator = dau_cycles_per_second / verilator_cycles_per_second

    return PerformanceDelta(
        dau_cycles_per_second=dau_cycles_per_second,
        vs_amaranth_ratio=vs_amaranth,
        vs_verilator_ratio=vs_verilator,
        multiplier_to_10x_current=10.0,
    )
