# CLI

dau-sim ships a [Typer](https://typer.tiangolo.com/)-based CLI for quick simulation and performance checks.

## Commands

```bash
dau-sim run-sv design.sv --top top_module --cycles 1000 --vcd out.vcd
dau-sim perf-sv design.sv --top top_module --cycles 30000 --repeats 3
```

`run-sv` executes a SystemVerilog design and prints the latest value of each signal after the requested number of cycles. Use `--vcd` to additionally write a VCD waveform file.

`perf-sv` composes and invokes the `task=tasks/analysis/perf-sv` ccflow task. It reports compile-time and simulation-time separately, along with node-separation diagnostics. CLI options populate the same task model fields that Hydra callers can override, so programmatic and command-line runs share configuration and result provenance.

The packaged task can also be composed directly:

```python
from dau_sim.config import run_request_config

result = run_request_config(
    "task",
    "tasks/analysis/perf-sv",
    model_values={"path": "design.sv", "top": "top_module", "cycles": 30000},
)
print(result.benchmark.cycles_per_second)
```
