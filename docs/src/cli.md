# CLI

dau-sim has a [Typer](https://typer.tiangolo.com/) CLI for quick simulations and performance checks.

## Commands

```bash
dau-sim run-sv design.sv --top top_module --cycles 1000 --vcd out.vcd
dau-sim perf-sv design.sv --top top_module --cycles 30000 --repeats 3
```

`run-sv` runs a SystemVerilog design and prints the final value of each signal after the requested number of cycles. Add `--vcd` to write a VCD waveform file as well. Like `perf-sv`, it composes a ccflow task, `task=tasks/sim/run-sv`, and the options fill that task's fields.

`perf-sv` composes and runs the `task=tasks/analysis/perf-sv` ccflow task. It reports compile time and simulation time separately, plus node-separation diagnostics. The CLI options fill the same task model fields that Hydra callers override, so command-line and programmatic runs share one configuration and one result type.

The packaged task can also be composed directly:

```python
from dau_sim.config import run_request_config

result = run_request_config(
    "task",
    "tasks/analysis/perf-sv",
    overrides=["model.path=design.sv", "model.top=top_module", "model.cycles=30000"],
)
print(result.benchmark.cycles_per_second)
```
