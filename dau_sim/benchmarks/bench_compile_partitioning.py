from __future__ import annotations

import pytest

from dau_sim.benchmarks.compile_partitioning import compile_partitioned_module


@pytest.mark.parametrize("n", [16, 64, 256, 1024])
def test_benchmark_compile_partitioning(benchmark, n: int) -> None:
    benchmark.pedantic(lambda: compile_partitioned_module(n), rounds=1, iterations=1)
