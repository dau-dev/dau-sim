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

- `from_amaranth()` — walks the Amaranth elaboration graph, lowering `Module`, `Signal`, and clock-domain assignments to IR nodes.
- `parse_sv()` / `parse_sv_file()` — uses pyslang to parse SystemVerilog/Verilog source and maps the resulting AST to IR nodes.
- IR can also be constructed directly for programmatic design generation.

### Intermediate Representation (IR)

A flat, typed graph of `Module`, `Signal`, `Port`, `ClockDomain`, `CombBlock`, `SeqBlock`, and expression/statement nodes. The IR is the single canonical form that all frontends target and all compiler passes consume.

Key files: `dau_sim/ir/module.py`, `dau_sim/ir/expr.py`, `dau_sim/ir/stmt.py`, `dau_sim/ir/types.py`.

### Compiler

`compile_module()` performs:

1. **Dependency analysis** — determines which combinational blocks depend on which signals (`depanalysis.py`).
1. **Code generation** — emits flat Python functions from IR statement/expression trees (`codegen.py`).
1. **Optimization** — selects the appropriate execution tier (interpreter, fast-tick, or batch no-trace) based on design characteristics.

### CSP Engine

The compiled design runs inside a [csp](https://github.com/Point72/csp) graph:

- Hardware signals → CSP time-series edges
- Combinational logic → CSP nodes (selective-settle: only re-evaluated when inputs change)
- Clock domains → CSP clock processes with correct posedge/negedge semantics

### Adapters

Post-simulation output:

- `write_vcd()` / `traces_to_vcd()` — IEEE 1364-2001 VCD waveform files
- FST and live streaming are planned
