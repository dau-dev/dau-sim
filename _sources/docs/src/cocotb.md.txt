# cocotb

dau-sim has a pure-Python [cocotb](https://www.cocotb.org/) backend, so a cocotb testbench that uses the triggers and handles below runs against it with no Verilog compilation and no external simulator. Testbenches that depend on simulator-specific features (VPI access beyond signal handles, `$dumpvars`, force/release from cocotb) are outside what the backend implements.

The backend implements Verilog non-blocking assignment (NBA) semantics, so `RisingEdge` callbacks see pre-NBA values, as they would in an HDL simulator.

cocotb is an optional dependency: `pip install "dau-sim[cocotb]"`.

## Running a testbench

```python
from dau_sim.frontends import from_amaranth
from dau_sim.backends.cocotb_backend import run_cocotb

from amaranth.hdl import Module
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class Counter(wiring.Component):
    en: In(1)
    count: Out(8)

    def elaborate(self, platform):
        m = Module()
        with m.If(self.en):
            m.d.sync += self.count.eq(self.count + 1)
        return m


run_cocotb(Counter(), test_module="test_counter")
```

You can also pass an IR `Module` directly instead of an Amaranth design.

## Writing the cocotb test

```python
# test_counter.py
import cocotb
from cocotb.clock import Clock
from cocotb._gpi_triggers import RisingEdge


@cocotb.test()
async def test_counting(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.en.value = 0
    await RisingEdge(dut.clk)

    dut.en.value = 1
    for expected in range(10):
        await RisingEdge(dut.clk)
        # NBA semantics: value visible one cycle after the edge
        await RisingEdge(dut.clk)
        assert int(dut.count.value) == expected + 1
```

## Semantic contracts

- **NBA ordering.** Value-change callbacks (`RisingEdge`, `FallingEdge`) observe pre-NBA values; sequential updates are staged, then applied.
- **Import order.** `cocotb.handle` must be imported before `cocotb._gpi_triggers` in patched simulator contexts. `run_cocotb` does this for you.
- **Multi-domain edges.** Posedge and negedge domains can share a clock and fire on the right edges.

## Stream contract monitoring

Wrap existing bench stimulus in the checker as an async context manager. It samples only clock edges outside reset, and on failure reports the interface prefix, the cycle and the rule that failed.

```python
from dau_sim.integrations.protocol import StreamContractMonitor


async with StreamContractMonitor(dut, dut.clk, "output_", reset=dut.rst, expected_batches=1):
    await drive_and_drain_one_batch(dut)
```

The default payload is `data` plus `last`. A stalled `valid` and its payload must stay stable until `ready`; a transfer is counted only when both are high. `expected_batches` lets the monitor decide whether a `last` is duplicated or missing, and rejects transfers after the final `last`.

`StatusContractMonitor` applies the same hold rule to `status_valid`, `status_error` and `status_error_code`. `mode="terminal"` requires one status per expected batch; `mode="mid_lane"` rejects success statuses. Both monitors are plain cocotb code with no simulator dependency, so one bench can use them under dau-sim or Verilator. Neither changes launcher or backend defaults.

## API

| Function / Class                       | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| `run_cocotb(design, test_module, ...)` | Run cocotb testbench against Amaranth design or IR `Module` |
| `SimulationEngine(module)`             | Low-level engine with NBA-correct event scheduling          |
| `StreamContractMonitor(...)`           | Check a prefixed valid/ready/data/last stream               |
| `StatusContractMonitor(...)`           | Check terminal or mid-lane status handshakes                |
