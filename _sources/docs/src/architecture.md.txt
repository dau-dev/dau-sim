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
1. **Execution** runs the design as one generated loop, or inside the CSP engine when it has memories or runs four-state.

### Generated run loop

Two-state designs without memories run as one generated Python function: signals in locals, clocks toggled at their half-periods, resets and sequential blocks inlined per domain, and each combinational component re-evaluated only when one of its external inputs changed in that tick. Traces are recorded on fired ticks.

### CSP Engine

Designs with memories, and four-state runs, execute inside a [csp](https://github.com/Point72/csp) graph: one tick node driving the compiled (or, for four-state, interpreted) sequential and combinational functions, memory ports in between, and traces as graph outputs.

### Adapters

Post-simulation output:

- `write_vcd()` and `traces_to_vcd()` produce IEEE 1364-2001 VCD waveform files.
- FST and live streaming are planned.
