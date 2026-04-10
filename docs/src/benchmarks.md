# Benchmarks

dau-sim includes a benchmark suite under `dau_sim/benchmarks/` that tracks simulation throughput across backends and measures internal compilation and evaluation performance. All numbers below were collected on an Apple M1 Pro (arm64, CPython 3.12) unless noted otherwise.

## Benchmark suite

### Cross-simulator comparison (`bench_cross_simulators.py`)

Compares dau-sim against Amaranth's native pysim simulator and Verilator on the same design: a 32-bit enabled counter running for a configurable number of cycles (set via `DAU_BENCH_CYCLES`, default 5,000).

Backends under test:

| Backend | What it measures |
|---|---|
| **dau-sim** | `from_amaranth` → `compile_module` → `cm.run(return_traces=False)` |
| **amaranth-sim** | Amaranth's built-in Python simulator (`Simulator` + generator process) |
| **cxxsim** | Amaranth's optional CXXRTL-backed simulator (skipped when unavailable) |
| **verilator-compile-run** | Full pipeline: write Verilog → `verilator --binary` → run executable |
| **verilator-runtime** | Pre-compiled Verilator binary execution only (compile cost excluded) |

Run the benchmark:

```bash
DAU_BENCH_CYCLES=100000 pytest dau_sim/benchmarks/bench_cross_simulators.py \
    --benchmark-only --benchmark-columns=mean,stddev,median
```

#### Results — 500k cycles

| Backend | Mean | vs Verilator runtime | vs Amaranth |
|---|---|---|---|
| verilator-runtime | 93 ms | 1.0× | 87× faster |
| dau-sim | 2.49 s | 27× slower | 3.2× faster |
| verilator-compile-run | 4.85 s | 52× slower | 1.7× faster |
| amaranth-sim | 8.08 s | 87× slower | 1.0× |

#### Results — 100k cycles

| Backend | Mean | vs Verilator runtime | vs Amaranth |
|---|---|---|---|
| verilator-runtime | 22.5 ms | 1.0× | 70× faster |
| dau-sim | 154 ms | 6.8× slower | 10× faster |
| amaranth-sim | 1,568 ms | 70× slower | 1.0× |
| verilator-compile-run | 4,509 ms | 200× slower | 2.9× faster |

Key observations:

- **dau-sim is 3–10× faster than Amaranth's pysim** across cycle counts, with the gap widening at higher counts due to lower per-tick overhead.
- **Verilator runtime is the throughput ceiling** — compiled C++ executing a simple counter at ~5M cycles/sec. dau-sim is 7–27× behind depending on cycle count.
- **Verilator compile+run is slower than dau-sim** for small-to-medium workloads because compilation dominates. dau-sim's zero-compile-step workflow gives it a significant advantage for iterative development.
- The Amaranth counter includes a reset signal (`rst`), which prevents dau-sim from using its fastest batch execution path. For reset-free IR designs, dau-sim achieves ~30 ms for 100k cycles (only 1.3× slower than Verilator runtime).

### Compile partitioning (`bench_compile_partitioning.py`)

Measures `compile_module` time as a function of the number of independent combinational blocks in a design. Tests N = 16, 64, 256, 1024 blocks, each a simple `assign o = a + const`. This benchmark validates that the dependency-analysis and block-partitioning phase scales well.

```bash
pytest dau_sim/benchmarks/bench_compile_partitioning.py --benchmark-only
```

### Selective settle (`bench_selective_settle.py`)

Measures runtime of a sequential design with N independent combinational components and configurable statements per component (1, 8, 32). Verifies that the selective-settle optimization — only re-evaluating combinational blocks whose inputs actually changed — keeps per-tick cost proportional to active components rather than total design size.

```bash
pytest dau_sim/benchmarks/bench_selective_settle.py --benchmark-only
```

## Execution modes and optimization tiers

dau-sim uses several execution strategies depending on the design and whether trace output is needed:

### CSP compiled path (default)

The compiler generates flat Python functions from the IR statement/expression trees (`dau_sim/compiler/codegen.py`) and executes them inside a single CSP node. Per-tick work:

1. Toggle clock signals at the correct half-period
2. Detect rising/falling edges via inlined comparisons
3. Execute the compiled sequential block for each fired domain
4. Re-evaluate affected combinational blocks (selective settle)
5. Optionally emit trace output

This path supports all designs including those with resets, combinational logic, and memories.

### Fast-tick path

For designs with **no combinational logic, no memories, and no resets**, the compiler generates a single `_fast_tick(S, clock_arr, tc)` function that inlines the clock toggle, edge detection, and sequential block body. This eliminates function-call overhead and changed-set tracking.

### Batch no-trace path

When `return_traces=False` and the design qualifies for fast-tick, dau-sim bypasses the CSP engine entirely and runs all ticks in a pure Python `for` loop. This eliminates ~400k CSP scheduling events for a 200k-tick simulation.

### Performance by execution mode (100k-cycle counter)

| Mode | Time | Speedup vs interpreter | Throughput |
|---|---|---|---|
| Interpreter (pre-optimization baseline) | 420 ms | 1.0× | 238k cycles/sec |
| CSP compiled (with fast-tick) | 71 ms | 5.9× | 1.4M cycles/sec |
| Batch no-trace | 30 ms | 13.9× | 3.3M cycles/sec |

## Running benchmarks

```bash
# Run all benchmarks
pytest dau_sim/benchmarks/ --benchmark-only

# Cross-simulator comparison with custom cycle count
DAU_BENCH_CYCLES=100000 pytest dau_sim/benchmarks/bench_cross_simulators.py --benchmark-only

# Save results to JSON for tracking
pytest dau_sim/benchmarks/bench_cross_simulators.py --benchmark-only \
    --benchmark-save=cross-runtime --benchmark-storage=dau_sim/benchmarks/results
```

Stored results live in `dau_sim/benchmarks/results/` for historical comparison.
