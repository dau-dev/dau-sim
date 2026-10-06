# Architecture

## Pipeline overview

```
Amaranth / SystemVerilog / Hand-built IR
                │
                ▼
        ┌───────────────┐
        │   Frontends   │  from_amaranth() / parse_sv()
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │      IR       │  Module, Signal, Expr, Stmt
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Compiler    │  compile_module() → CompiledModule
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  CSP Engine   │  cm.run() → traces
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   Adapters    │  write_vcd() / traces_to_vcd()
        └───────────────┘
```

## Layers

### Frontends

Convert external design representations into the dau-sim IR:

- `from_amaranth()` walks the Amaranth elaboration graph and lowers `Module`, `Signal` and clock-domain assignments to IR nodes.
- `parse_sv()` and `parse_sv_file()` parse SystemVerilog or Verilog source with pyslang and map the resulting AST to IR nodes.
- The IR can also be built directly, for programmatic design generation.

### Intermediate Representation (IR)

A flat, typed graph of `Module`, `Signal`, `Port`, `ClockDomain`, `CombBlock`, `SeqBlock` and expression and statement nodes. Every frontend produces this form and every compiler pass consumes it.

Key files: `dau_sim/ir/module.py`, `dau_sim/ir/expr.py`, `dau_sim/ir/stmt.py`, `dau_sim/ir/types.py`.

### Compiler

`compile_module()` performs:

1. **Dependency analysis** works out which combinational blocks depend on which signals (`depanalysis.py`).
1. **Code generation** emits flat Python functions from the IR statement and expression trees (`codegen.py`).
1. **Optimization** picks the execution tier (interpreter, fast-tick, or batch no-trace) that the design qualifies for.

### CSP Engine

The compiled design runs inside a [csp](https://github.com/Point72/csp) graph:

- Hardware signals are CSP time-series edges.
- Combinational logic is a set of CSP nodes, re-evaluated only when an input changes (selective settle).
- Clock domains are CSP clock processes with posedge and negedge semantics.

### Adapters

Post-simulation output:

- `write_vcd()` and `traces_to_vcd()` produce IEEE 1364-2001 VCD waveform files.
- FST and live streaming are planned.
