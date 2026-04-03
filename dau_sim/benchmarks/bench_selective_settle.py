from __future__ import annotations

import pytest

from dau_sim.benchmarks.selective_settle import run_partitioned_seq


@pytest.mark.parametrize("n", [16, 64, 256, 512])
@pytest.mark.parametrize("stmts_per_component", [1, 8, 32])
def test_benchmark_selective_settle(benchmark, n: int, stmts_per_component: int) -> None:
    benchmark.pedantic(lambda: run_partitioned_seq(n, stmts_per_component, cycles=200), rounds=1, iterations=1)
