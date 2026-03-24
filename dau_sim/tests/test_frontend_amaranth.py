import pytest
from amaranth.hdl import Cat, ClockDomain, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from dau_sim.compiler import compile_module
from dau_sim.frontends import from_amaranth
from dau_sim.ir import (
    BinaryOp,
    EdgePolarity,
    PortDirection,
    ResetStyle,
    Shape,
    UnaryOp,
)
from dau_sim.ir.expr import Binary, Concat, Const as IRConst, Mux as IRMux, SignalRef, Slice, Unary
from dau_sim.ir.stmt import Assign, Switch


def _port_names(mod, direction=None):
    """Return port names from an IR module, optionally filtered by direction."""
    if direction is None:
        return {p.name for p in mod.ports}
    return {p.name for p in mod.ports if p.direction == direction}


def _port_shape(mod, name):
    """Return the shape for a named port."""
    for p in mod.ports:
        if p.name == name:
            return p.shape
    raise KeyError(name)


def _signal_names(mod):
    """Return internal signal names."""
    return {s.name for s in mod.signals}


class TestModuleExtraction:
    """Test port/signal/domain extraction from Amaranth designs."""

    def test_component_ports(self):
        """Component ports should be extracted with correct direction."""

        class Adder(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a + self.b)
                return m

        mod = from_amaranth(Adder())
        assert mod.name == "Adder"
        assert _port_names(mod, PortDirection.INPUT) >= {"a", "b"}
        assert _port_names(mod, PortDirection.OUTPUT) == {"y"}

    def test_component_wide_ports(self):
        """Wide port shapes should be preserved."""

        class Wide(Component):
            data_in: In(16)
            data_out: Out(32)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.data_out.eq(self.data_in)
                return m

        mod = from_amaranth(Wide())
        assert _port_shape(mod, "data_in") == Shape(16, False)
        assert _port_shape(mod, "data_out") == Shape(32, False)

    def test_component_signed_port(self):
        """Signed shapes should be preserved."""

        class Signed(Component):
            a: In(signed(8))
            y: Out(signed(8))

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a)
                return m

        mod = from_amaranth(Signed())
        assert _port_shape(mod, "a") == Shape(8, True)
        assert _port_shape(mod, "y") == Shape(8, True)

    def test_bare_elaboratable(self):
        """Bare Elaboratable produces a module with internal signals only."""

        class Bare(Elaboratable):
            def __init__(self):
                self.a = Signal(4)
                self.b = Signal(4)
                self.y = Signal(4)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a & self.b)
                return m

        mod = from_amaranth(Bare())
        assert mod.name == "Bare"
        # No formal ports — all signals are internal
        assert all(p.direction == PortDirection.INPUT for p in mod.ports)  # only clk/rst if any
        # Internal signals include a, b, y
        all_names = _signal_names(mod) | _port_names(mod)
        assert {"a", "b", "y"} <= all_names

    def test_custom_name(self):
        """Custom module name overrides class name."""

        class Foo(Component):
            x: In(1)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.x)
                return m

        mod = from_amaranth(Foo(), name="my_module")
        assert mod.name == "my_module"

    def test_signal_init_values(self):
        """Signal init values should be preserved."""

        class WithInit(Component):
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                counter = Signal(8, init=42)
                m.d.sync += counter.eq(counter + 1)
                m.d.comb += self.y.eq(counter)
                return m

        mod = from_amaranth(WithInit())
        # Find the counter signal and check init
        counter_sigs = [s for s in mod.signals if s.name == "counter"]
        assert len(counter_sigs) == 1
        assert counter_sigs[0].init == 42


class TestExpressionLowering:
    """Test lowering of Amaranth AST expressions to IR."""

    def _get_comb_assign(self, design):
        """Extract the first comb assign from a design."""
        mod = from_amaranth(design)
        assert len(mod.comb_blocks) >= 1
        for stmt in mod.comb_blocks[0].stmts:
            if isinstance(stmt, Assign):
                return stmt
        raise AssertionError("No Assign found in comb block")

    def test_constant(self):
        class C(Component):
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(42)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, IRConst)
        assert stmt.value.value == 42

    def test_add(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a + self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.ADD

    def test_sub(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a - self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.SUB

    def test_mul(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(16)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a * self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.MUL

    def test_bitwise_and(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a & self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.AND

    def test_bitwise_or(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a | self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.OR

    def test_bitwise_xor(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a ^ self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.XOR

    def test_shift_left(self):
        class C(Component):
            a: In(8)
            y: Out(16)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a << 2)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.SHL

    def test_shift_right(self):
        class C(Component):
            a: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a >> 2)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.SHR

    def test_eq_comparison(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a == self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.EQ

    def test_ne_comparison(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a != self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.NE

    def test_lt_comparison(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a < self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.LT

    def test_ge_comparison(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a >= self.b)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Binary)
        assert stmt.value.op == BinaryOp.GE

    def test_bitwise_not(self):
        class C(Component):
            a: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(~self.a)
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Unary)
        assert stmt.value.op == UnaryOp.NOT

    def test_mux(self):
        class C(Component):
            sel: In(1)
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(Mux(self.sel, self.a, self.b))
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, IRMux)

    def test_slice(self):
        class C(Component):
            a: In(8)
            y: Out(4)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a[0:4])
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Slice)
        assert stmt.value.low == 0
        assert stmt.value.high == 4
        assert stmt.value.shape == Shape(4, False)

    def test_concat(self):
        """Cat(a, b) should produce Concat with reversed part order (MSB first)."""

        class C(Component):
            a: In(4)
            b: In(4)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(Cat(self.a, self.b))
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Concat)
        assert len(stmt.value.parts) == 2
        # MSB first in IR: parts[0] should be 'b', parts[1] should be 'a'
        assert isinstance(stmt.value.parts[0], SignalRef)
        assert stmt.value.parts[0].name == "b"
        assert isinstance(stmt.value.parts[1], SignalRef)
        assert stmt.value.parts[1].name == "a"

    def test_any_operator(self):
        """Signal.any() produces reduce-OR (r|) → mapped to RED_OR."""

        class C(Component):
            a: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a.any())
                return m

        stmt = self._get_comb_assign(C())
        assert isinstance(stmt.value, Unary)
        assert stmt.value.op == UnaryOp.RED_OR


class TestStatementLowering:
    """Test lowering of Amaranth statements to IR."""

    def test_assign(self):
        class C(Component):
            a: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a)
                return m

        mod = from_amaranth(C())
        assert len(mod.comb_blocks) >= 1
        stmts = mod.comb_blocks[0].stmts
        assigns = [s for s in stmts if isinstance(s, Assign)]
        assert len(assigns) >= 1
        assert assigns[0].target == "y"

    def test_if_else_becomes_switch(self):
        """m.If/m.Else is lowered to Switch in Amaranth AST."""

        class C(Component):
            cond: In(1)
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                with m.If(self.cond):
                    m.d.comb += self.y.eq(self.a)
                with m.Else():
                    m.d.comb += self.y.eq(self.b)
                return m

        mod = from_amaranth(C())
        stmts = mod.comb_blocks[0].stmts
        switches = [s for s in stmts if isinstance(s, Switch)]
        assert len(switches) == 1
        sw = switches[0]
        # Should have a pattern=1 case and a default (None) case
        patterns = [c[0] for c in sw.cases]
        assert 1 in patterns
        assert None in patterns

    def test_switch(self):
        """m.Switch with explicit Case values."""

        class C(Component):
            sel: In(2)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                with m.Switch(self.sel):
                    with m.Case(0):
                        m.d.comb += self.y.eq(10)
                    with m.Case(1):
                        m.d.comb += self.y.eq(20)
                    with m.Case(2):
                        m.d.comb += self.y.eq(30)
                    with m.Default():
                        m.d.comb += self.y.eq(0)
                return m

        mod = from_amaranth(C())
        stmts = mod.comb_blocks[0].stmts
        switches = [s for s in stmts if isinstance(s, Switch)]
        assert len(switches) == 1
        sw = switches[0]
        patterns = {c[0] for c in sw.cases}
        assert {0, 1, 2, None} == patterns

    def test_sequential_assign(self):
        """Sequential domain assign produces SeqBlock."""

        class C(Component):
            d: In(8)
            q: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.q.eq(self.d)
                return m

        mod = from_amaranth(C())
        assert len(mod.seq_blocks) >= 1
        sb = mod.seq_blocks[0]
        assert sb.domain == "sync"
        assigns = [s for s in sb.stmts if isinstance(s, Assign)]
        assert len(assigns) >= 1
        assert assigns[0].target == "q"


class TestClockDomainMapping:
    """Test clock domain extraction and auto-creation."""

    def test_default_sync_domain(self):
        """Using m.d.sync should auto-create a 'sync' posedge domain."""

        class C(Component):
            q: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.q.eq(self.q + 1)
                return m

        mod = from_amaranth(C())
        assert len(mod.clock_domains) == 1
        cd = mod.clock_domains[0]
        assert cd.name == "sync"
        assert cd.clk == "clk"
        assert cd.rst == "rst"
        assert cd.edge == EdgePolarity.POSEDGE
        assert cd.rst_style == ResetStyle.SYNC

    def test_clk_rst_as_input_ports(self):
        """Auto-created clock/reset should appear as input ports."""

        class C(Component):
            q: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.q.eq(self.q + 1)
                return m

        mod = from_amaranth(C())
        port_map = {p.name: p.direction for p in mod.ports}
        assert port_map.get("clk") == PortDirection.INPUT
        assert port_map.get("rst") == PortDirection.INPUT

    def test_comb_only_no_clock_domain(self):
        """Purely combinational design should have no clock domains."""

        class C(Component):
            a: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a)
                return m

        mod = from_amaranth(C())
        assert len(mod.clock_domains) == 0

    def test_explicit_domain(self):
        """Explicitly added clock domain should be extracted properly."""

        class C(Component):
            q: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.domains += ClockDomain("fast")
                m.d.fast += self.q.eq(self.q + 1)
                return m

        mod = from_amaranth(C())
        domain_names = {d.name for d in mod.clock_domains}
        assert "fast" in domain_names
        fast = next(d for d in mod.clock_domains if d.name == "fast")
        assert fast.clk == "fast_clk"
        assert fast.rst == "fast_rst"


class TestEndToEndCombinational:
    """Combinational designs: Amaranth → compile → simulate."""

    def test_adder(self):
        class Adder(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a + self.b)
                return m

        mod = from_amaranth(Adder())
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 10, "b": 20})
        assert [v for _, v in traces["y"]][-1] == 30

    def test_adder_overflow(self):
        """8-bit add wraps at 256."""

        class Adder(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a + self.b)
                return m

        mod = from_amaranth(Adder())
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 200, "b": 100})
        # 200 + 100 = 300, masked to 8-bit = 44
        assert [v for _, v in traces["y"]][-1] == 44

    def test_mux(self):
        class M(Component):
            sel: In(1)
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(Mux(self.sel, self.a, self.b))
                return m

        mod = from_amaranth(M())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"sel": 1, "a": 42, "b": 99})
        assert [v for _, v in t["y"]][-1] == 42
        t = cm.run(cycles=1, inputs={"sel": 0, "a": 42, "b": 99})
        assert [v for _, v in t["y"]][-1] == 99

    def test_bitwise_ops(self):
        class B(Component):
            a: In(8)
            b: In(8)
            y_and: Out(8)
            y_or: Out(8)
            y_xor: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += [
                    self.y_and.eq(self.a & self.b),
                    self.y_or.eq(self.a | self.b),
                    self.y_xor.eq(self.a ^ self.b),
                ]
                return m

        mod = from_amaranth(B())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"a": 0xAA, "b": 0x55})
        assert [v for _, v in t["y_and"]][-1] == 0x00
        assert [v for _, v in t["y_or"]][-1] == 0xFF
        assert [v for _, v in t["y_xor"]][-1] == 0xFF

    def test_invert(self):
        class Inv(Component):
            a: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(~self.a)
                return m

        mod = from_amaranth(Inv())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"a": 0xAA})
        assert [v for _, v in t["y"]][-1] == 0x55

    def test_slice(self):
        class S(Component):
            a: In(8)
            y: Out(4)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a[0:4])
                return m

        mod = from_amaranth(S())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"a": 0xAB})
        # Lower 4 bits of 0xAB = 0xB = 11
        assert [v for _, v in t["y"]][-1] == 0xB

    def test_concat(self):
        class C(Component):
            lo: In(4)
            hi: In(4)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(Cat(self.lo, self.hi))
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"lo": 0xA, "hi": 0x5})
        # Cat(lo, hi) = hi:lo = 0x5A
        assert [v for _, v in t["y"]][-1] == 0x5A

    def test_if_else(self):
        class C(Component):
            cond: In(1)
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                with m.If(self.cond):
                    m.d.comb += self.y.eq(self.a)
                with m.Else():
                    m.d.comb += self.y.eq(self.b)
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"cond": 1, "a": 10, "b": 20})
        assert [v for _, v in t["y"]][-1] == 10
        t = cm.run(cycles=1, inputs={"cond": 0, "a": 10, "b": 20})
        assert [v for _, v in t["y"]][-1] == 20

    def test_switch_case(self):
        class C(Component):
            sel: In(2)
            y: Out(8)

            def elaborate(self, platform):
                m = Module()
                with m.Switch(self.sel):
                    with m.Case(0):
                        m.d.comb += self.y.eq(10)
                    with m.Case(1):
                        m.d.comb += self.y.eq(20)
                    with m.Case(2):
                        m.d.comb += self.y.eq(30)
                    with m.Default():
                        m.d.comb += self.y.eq(0)
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        for sel_val, expected in [(0, 10), (1, 20), (2, 30), (3, 0)]:
            t = cm.run(cycles=1, inputs={"sel": sel_val})
            actual = [v for _, v in t["y"]][-1]
            assert actual == expected, f"sel={sel_val}: expected {expected}, got {actual}"

    def test_comparison_eq(self):
        class C(Component):
            a: In(8)
            b: In(8)
            y: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a == self.b)
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"a": 42, "b": 42})
        assert [v for _, v in t["y"]][-1] == 1
        t = cm.run(cycles=1, inputs={"a": 42, "b": 43})
        assert [v for _, v in t["y"]][-1] == 0

    def test_shift_left(self):
        class C(Component):
            a: In(8)
            y: Out(16)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a << 4)
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        t = cm.run(cycles=1, inputs={"a": 0x0F})
        assert [v for _, v in t["y"]][-1] == 0xF0


class TestEndToEndSequential:
    """Sequential designs: Amaranth → compile → simulate."""

    def test_counter(self):
        """Simple incrementing counter."""

        class Counter(Component):
            count: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.count.eq(self.count + 1)
                return m

        mod = from_amaranth(Counter())
        cm = compile_module(mod)
        traces = cm.run(cycles=5, inputs={})
        vals = [v for _, v in traces["count"]]
        assert vals == [1, 2, 3, 4, 5]

    def test_counter_wraps(self):
        """4-bit counter wraps at 16."""

        class Counter4(Component):
            count: Out(4)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.count.eq(self.count + 1)
                return m

        mod = from_amaranth(Counter4())
        cm = compile_module(mod)
        traces = cm.run(cycles=18, inputs={})
        vals = [v for _, v in traces["count"]]
        # Should wrap: ..., 14, 15, 0, 1
        assert vals[14] == 15
        assert vals[15] == 0
        assert vals[16] == 1

    def test_dff(self):
        """D flip-flop: q captures d on each clock edge."""

        class DFF(Component):
            d: In(8)
            q: Out(8)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.q.eq(self.d)
                return m

        mod = from_amaranth(DFF())
        cm = compile_module(mod)
        traces = cm.run(cycles=3, inputs={"d": 42})
        vals = [v for _, v in traces["q"]]
        assert all(v == 42 for v in vals)

    def test_shift_register(self):
        """4-bit serial-in shift register."""

        class SR(Component):
            d_in: In(1)
            q: Out(4)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.q.eq(Cat(self.d_in, self.q[0:3]))
                return m

        mod = from_amaranth(SR())
        cm = compile_module(mod)
        # Shift in 1s for 4 cycles
        traces = cm.run(cycles=4, inputs={"d_in": 1})
        vals = [v for _, v in traces["q"]]
        # After each cycle, one more bit is 1 (from LSB):
        # Cycle 1: Cat(1, q[0:3]=000) = 0001 = 1
        # Cycle 2: Cat(1, q[0:3]=001) = 0011 = 3
        # Cycle 3: Cat(1, q[0:3]=011) = 0111 = 7
        # Cycle 4: Cat(1, q[0:3]=111) = 1111 = 15
        assert vals == [1, 3, 7, 15]

    def test_counter_with_if_reset(self):
        """Counter with conditional reset using m.If."""

        class CR(Component):
            en: In(1)
            count: Out(8)

            def elaborate(self, platform):
                m = Module()
                with m.If(self.en):
                    m.d.sync += self.count.eq(self.count + 1)
                return m

        mod = from_amaranth(CR())
        cm = compile_module(mod)
        # Enable counting
        traces = cm.run(cycles=3, inputs={"en": 1})
        vals = [v for _, v in traces["count"]]
        assert vals == [1, 2, 3]
        # Disabled — count stays at 0
        traces = cm.run(cycles=3, inputs={"en": 0})
        vals = [v for _, v in traces["count"]]
        assert all(v == 0 for v in vals)

    def test_mixed_comb_seq(self):
        """Counter with combinational output decode."""

        class C(Component):
            count: Out(4)
            is_max: Out(1)

            def elaborate(self, platform):
                m = Module()
                m.d.sync += self.count.eq(self.count + 1)
                m.d.comb += self.is_max.eq(self.count == 15)
                return m

        mod = from_amaranth(C())
        cm = compile_module(mod)
        traces = cm.run(cycles=16, inputs={})
        count_vals = [v for _, v in traces["count"]]
        is_max_vals = [v for _, v in traces["is_max"]]
        assert count_vals[14] == 15
        assert is_max_vals[14] == 1
        # After wrap (on cycle 15, count becomes 0)
        assert count_vals[15] == 0
        assert is_max_vals[15] == 0


class TestErrorHandling:
    """Test error paths."""

    def test_unsupported_part_raises(self):
        """Dynamic Part node should raise NotImplementedError."""

        class C(Component):
            a: In(8)
            idx: In(2)
            y: Out(2)

            def elaborate(self, platform):
                m = Module()
                m.d.comb += self.y.eq(self.a.word_select(self.idx, 2))
                return m

        with pytest.raises(NotImplementedError, match="Part"):
            from_amaranth(C())
