# Benchmarks

The benchmark suite under `dau_sim/benchmarks/` tracks simulation throughput across backends and measures compile and evaluation cost inside dau-sim. All numbers below were collected on an Apple M5 Pro (arm64, CPython 3.12) unless noted otherwise.

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
| dau-sim               | 48 ms  | 1.16× faster         | 85× faster  |
| verilator-runtime     | 56 ms  | 1.0×                 | 73× faster  |
| verilator-compile-run | 3.09 s | 55× slower           | 1.3× faster |
| amaranth-sim          | 4.08 s | 73× slower           | 1.0×        |

The 500k-cycle table is the run recorded in
`dau_sim/benchmarks/results/cross-runtime-500k.json` (2026-10-08, one laptop,
machine details in the file). dau-sim runs with `return_traces=False` here;
with traces on, the same run takes about 2.5× longer (the list appends per
fired tick). The design is a 32-bit counter, the smallest sequential design
there is; on larger designs the per-statement cost of Python widens the gap
again. Rerun the suite on your machine before comparing
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

## Execution modes

dau-sim picks an execution strategy from the design:

### Generated run loop (default)

For two-state designs without memories, the compiler generates one Python function for the whole run (`CodeGen.build_run_loop` in `dau_sim/compiler/codegen.py`). Every signal lives in a Python local for the duration of the loop and is written back at the end, so a tick costs a handful of local operations and no function calls. Each tick, in order:

1. Toggles each clock at its half-period and notes which domains fired
1. Applies an active asynchronous reset (init values, and the domain does not fire)
1. Runs the sequential block, or the synchronous reset, of each domain that fired
1. Re-evaluates each combinational component whose external inputs changed since the tick began
1. If a domain fired and traces were asked for, records the traced signals

The loop covers resets, combinational logic and `$finish`. Traces cost one list append per traced signal per fired tick; `return_traces=False` removes them.

### CSP engine

Designs with memories, and four-state (X/Z) runs, execute inside a [csp](https://github.com/Point72/csp) graph: one node per tick, with codegen-compiled sequential and combinational functions (two-state) or the tree-walking evaluator (four-state), memory ports evaluated between the sequential blocks and the combinational settle, and traces as graph outputs.

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
