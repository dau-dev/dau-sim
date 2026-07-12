# Amaranth

dau-sim accepts any [Amaranth HDL](https://amaranth-lang.org/) `Elaboratable` or `Component` via `from_amaranth()`, which lowers it to the dau-sim IR before compilation.

## Basic example — counter

```python
from amaranth.hdl import Module
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from dau_sim.frontends import from_amaranth
from dau_sim.compiler import compile_module


class Counter(wiring.Component):
    en: In(1)
    count: Out(8)

    def elaborate(self, platform):
        m = Module()
        with m.If(self.en):
            m.d.sync += self.count.eq(self.count + 1)
        return m


# Lower to IR, compile, and simulate
ir_module = from_amaranth(Counter())
cm = compile_module(ir_module)
traces = cm.run(cycles=20, inputs={"en": 1})

# Write VCD waveform file
cm.write_vcd("counter.vcd", traces)

# Or get the VCD as a string
vcd_str = cm.traces_to_vcd(traces)
```

## Sequential design with clock domains

```python
from amaranth.hdl import Cat, Module, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from datetime import timedelta

from dau_sim.frontends import from_amaranth
from dau_sim.compiler import compile_module


class ShiftRegister(wiring.Component):
    d_in: In(1)
    d_out: Out(1)

    def elaborate(self, platform):
        m = Module()
        reg = Signal(4)
        m.d.sync += reg.eq(Cat(self.d_in, reg[:-1]))
        m.d.comb += self.d_out.eq(reg[-1])
        return m


ir = from_amaranth(ShiftRegister())
cm = compile_module(ir)

# Run with a custom clock period
traces = cm.run(
    cycles=10,
    clock_period=timedelta(microseconds=1),
    inputs={"d_in": 1},
)

cm.write_vcd("shift_reg.vcd", traces, timescale="1ns")
```

## Notes

- Any Amaranth `wiring.Component` or plain `Elaboratable` is accepted.
- Clock domains are inferred from `m.d.<domain>` usage; `sync` maps to a default 1 MHz clock that can be overridden with `clock_period`.
- See the [API reference](api.md) for the full `from_amaranth` signature.
