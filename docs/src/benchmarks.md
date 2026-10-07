# Benchmarks

The benchmark suite under `dau_sim/benchmarks/` tracks simulation throughput across backends and measures compile and evaluation cost inside dau-sim. All numbers below were collected on an Apple M1 Pro (arm64, CPython 3.12) unless noted otherwise.

## Benchmark suite

### Cross-simulator comparison (`bench_cross_simulators.py`)

Compares dau-sim with Amaranth's pysim simulator and with Verilator on the same design: a 32-bit counter with an enable, run for a configurable number of cycles (`DAU_BENCH_CYCLES`, default 5,000).

Backends under test:

| Backend                   | What it measures                                                           |
| ------------------------- | -------------------------------------------------------------------------- |
| **dau-sim**               | `from_amaranth`, then `compile_module`, then `cm.run(return_traces=False)` |
| **amaranth-sim**          | Amaranth's built-in Python simulator (`Simulator` + generator process)     |
| **cxxsim**                | Amaranth's optional CXXRTL-backed simulator (skipped when unavailable)     |
| **verilator-compile-run** | Full pipeline: write Verilog, `verilator --binary`, run the executable     |
| **verilator-runtime**     | Pre-compiled Verilator binary execution only (compile cost excluded)       |

Run the benchmark:

```bash
DAU_BENCH_CYCLES=100000 pytest dau_sim/benchmarks/bench_cross_simulators.py \
    --benchmark-only --benchmark-columns=mean,stddev,median
```

#### Results at 500k cycles

| Backend               | Mean   | vs Verilator runtime | vs Amaranth |
| --------------------- | ------ | -------------------- | ----------- |
| verilator-runtime     | 93 ms  | 1.0×                 | 87× faster  |
| dau-sim               | 2.49 s | 27× slower           | 3.2× faster |
| verilator-compile-run | 4.85 s | 52× slower           | 1.7× faster |
| amaranth-sim          | 8.08 s | 87× slower           | 1.0×        |

The 500k-cycle table is the run recorded in
`dau_sim/benchmarks/results/cross-runtime-500k.json` (2026-04-03, one laptop,
machine details in the file). Rerun the suite on your machine before comparing
against it; the ratios move with the host.

### Compile partitioning (`bench_compile_partitioning.py`)

Measures `compile_module` time against the number of independent combinational blocks in a design, for N = 16, 64, 256 and 1024 blocks, each a simple `assign o = a + const`. It checks that dependency analysis and block partitioning scale.

```bash
pytest dau_sim/benchmarks/bench_compile_partitioning.py --benchmark-only
```

### Selective settle (`bench_selective_settle.py`)

Measures the runtime of a sequential design with N independent combinational components and a configurable number of statements per component (1, 8, 32). It checks that selective settle, which re-evaluates only the combinational blocks whose inputs changed, keeps per-tick cost proportional to the number of active components rather than the size of the design.

```bash
pytest dau_sim/benchmarks/bench_selective_settle.py --benchmark-only
```

## Execution modes and optimization tiers

dau-sim picks an execution strategy from the design and from whether you asked for traces:

### CSP compiled path (default)

The compiler generates flat Python functions from the IR statement and expression trees (`dau_sim/compiler/codegen.py`) and runs them inside a single CSP node. Each tick:

1. Toggles clock signals at their half-period
1. Detects rising and falling edges with inlined comparisons
1. Runs the compiled sequential block for each domain that fired
1. Re-evaluates the affected combinational blocks (selective settle)
1. Emits trace output if requested

This path handles the designs the frontends lower, including those with resets, combinational logic and memories.

### Fast-tick path

For designs with **no combinational logic, no memories and no resets**, the compiler generates a single `_fast_tick(S, clock_arr, tc)` function that inlines the clock toggle, edge detection and sequential block body. There is no function-call overhead and no changed-set tracking.

### Batch no-trace path

When `return_traces=False` and the design qualifies for fast-tick, dau-sim skips the CSP engine and runs every tick in a plain Python `for` loop. For a 200k-tick simulation that removes about 400k CSP scheduling events.

The batch path skips the CSP engine entirely, so for a qualifying design its
cost is the generated tick function alone; the saving over the compiled path is
the engine's per-tick overhead. Measure it with `perf-sv` on the design you
care about rather than taking a number from here.

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

Saved results are kept in `dau_sim/benchmarks/results/` for comparison over time.
