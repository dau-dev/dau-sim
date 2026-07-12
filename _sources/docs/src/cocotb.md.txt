# cocotb

dau-sim includes a pure-Python [cocotb](https://www.cocotb.org/) backend that lets you run existing cocotb testbenches directly — no Verilog compilation or external simulator required.

The backend implements Verilog-style non-blocking assignment (NBA) semantics so that `RisingEdge` callbacks see pre-NBA values, matching real HDL simulator behavior.

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
    clock.start()

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

- **NBA-correct ordering** — value-change callbacks (`RisingEdge`/`FallingEdge`) observe pre-NBA values; sequential updates are staged then applied.
- **Import order** — `cocotb.handle` must be imported before `cocotb._gpi_triggers` in patched simulator contexts. `run_cocotb` handles this automatically.
- **Multi-domain edge semantics** — posedge and negedge domains can share clocks and progress with correct edge-firing behavior.

## API

| Function / Class                       | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| `run_cocotb(design, test_module, ...)` | Run cocotb testbench against Amaranth design or IR `Module` |
| `SimulationEngine(module)`             | Low-level engine with NBA-correct event scheduling          |
