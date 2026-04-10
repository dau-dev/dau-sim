# API Reference

## High-level interactive API

```python
from dau_sim import Simulator

sim = Simulator.from_sv_file("design.sv", top="top_module")
result = sim.run(cycles=1000, inputs={"en": 1})

print(result.latest("count"))
rows = result.to_rows(signals=["count"])
sim.write_vcd("out.vcd", result)
```

## Frontends

| Function                               | Description                                           |
| -------------------------------------- | ----------------------------------------------------- |
| `from_amaranth(elaboratable)`          | Lower an Amaranth `Elaboratable` or `Component` to IR |
| `parse_sv(source, top_module=None)`    | Parse SystemVerilog source string to IR               |
| `parse_sv_file(path, top_module=None)` | Parse SystemVerilog file to IR                        |
| `from_dau_build(mod)`                  | Bridge from a `dau_build.Module` to IR                |

## Compiler

| Function / Method                              | Description                            |
| ---------------------------------------------- | -------------------------------------- |
| `compile_module(module, four_state=False)`     | Compile IR `Module` → `CompiledModule` |
| `cm.run(cycles, clock_period, inputs, clocks)` | Simulate and return traces             |
| `cm.write_vcd(path, traces, timescale="1ns")`  | Write traces to VCD file               |
| `cm.traces_to_vcd(traces, timescale="1ns")`    | Convert traces to VCD string           |

## Backends

| Function / Class                       | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| `run_cocotb(design, test_module, ...)` | Run cocotb testbench against Amaranth design or IR `Module` |
| `SimulationEngine(module)`             | Low-level engine with NBA-correct event scheduling          |

## IR types

| Type                                                              | Description                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
| `Module`                                                          | Top-level container with ports, signals, clock domains, blocks |
| `Signal`                                                          | Named bit-vector with shape and initial value                  |
| `Port`                                                            | Signal with direction (`INPUT`/`OUTPUT`/`INOUT`)               |
| `Shape(width, signed)`                                            | Bit-width and signedness                                       |
| `ClockDomain`                                                     | Clock signal, edge polarity, optional reset                    |
| `CombBlock` / `SeqBlock`                                          | Combinational / sequential logic blocks                        |
| `Assign`, `IfElse`, `Switch`                                      | Statement types                                                |
| `Const`, `SignalRef`, `Binary`, `Unary`, `Mux`, `Concat`, `Slice` | Expression types                                               |
