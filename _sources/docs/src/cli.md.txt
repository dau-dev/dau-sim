# CLI

dau-sim ships a [Typer](https://typer.tiangolo.com/)-based CLI for quick simulation and performance checks.

## Commands

```bash
dau-sim run-sv design.sv --top top_module --cycles 1000 --vcd out.vcd
dau-sim perf-sv design.sv --top top_module --cycles 30000 --repeats 3
```

`run-sv` executes a SystemVerilog design and prints the latest value of each signal after the requested number of cycles. Use `--vcd` to additionally write a VCD waveform file.

`perf-sv` runs the same pipeline but reports compile-time and simulation-time separately, along with node-separation diagnostics. This helps identify where non-C++ optimization effort should be focused.
