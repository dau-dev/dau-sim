# Verilog / SystemVerilog

dau-sim parses SystemVerilog and Verilog source via [pyslang](https://github.com/MikePopoloski/pyslang), then lowers the AST to the dau-sim IR.

## Simulate from source string

```python
from dau_sim.frontends import parse_sv
from dau_sim.compiler import compile_module

ir_module = parse_sv("""
    module adder(
        input  logic [7:0] a, b,
        output logic [7:0] y
    );
        assign y = a + b;
    endmodule
""", top_module="adder")

cm = compile_module(ir_module)
traces = cm.run(cycles=1, inputs={"a": 100, "b": 55})

# traces["y"] contains [(timestamp, 155)]
```

## Simulate from file

```python
from dau_sim.frontends import parse_sv_file
from dau_sim.compiler import compile_module

ir_module = parse_sv_file("design.sv", top_module="adder")
cm = compile_module(ir_module)
traces = cm.run(cycles=10, inputs={"a": 42, "b": 10})
```

## Hand-constructed IR

For programmatic design generation you can build the IR directly without a frontend:

```python
from dau_sim.ir import *
from dau_sim.compiler import compile_module

# Build a 4-bit AND gate from IR primitives
a = Signal("a", Shape(4))
b = Signal("b", Shape(4))
y = Signal("y", Shape(4))

mod = Module(
    name="and_gate",
    ports=(
        Port(a, PortDirection.INPUT),
        Port(b, PortDirection.INPUT),
        Port(y, PortDirection.OUTPUT),
    ),
    signals=(),
    clock_domains=(),
    comb_blocks=(
        CombBlock(stmts=(
            Assign("y", Binary(Shape(4), BinaryOp.AND, SignalRef(Shape(4), "a"), SignalRef(Shape(4), "b"))),
        )),
    ),
    seq_blocks=(),
)

cm = compile_module(mod)
traces = cm.run(cycles=1, inputs={"a": 0b1100, "b": 0b1010})
# traces["y"] -> [(timestamp, 0b1000)]
```

See the [API reference](api.md) for all IR node types.
