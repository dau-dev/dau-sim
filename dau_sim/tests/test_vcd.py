import os
import tempfile
from datetime import datetime, timedelta

import pytest

from dau_sim.adapters.vcd import (
    _format_value,
    _make_id,
    _parse_timescale_ns,
    traces_to_vcd,
    write_vcd,
)
from dau_sim.compiler import compile_module
from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    EdgePolarity,
    Module,
    Port,
    PortDirection,
    ResetStyle,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
)


def _parse_vcd_signals(vcd: str) -> dict[str, dict]:
    """Parse VCD variable declarations.

    Returns: {name: {"id": str, "width": int, "type": str}}
    """
    signals = {}
    for line in vcd.splitlines():
        line = line.strip()
        if line.startswith("$var"):
            parts = line.split()
            # $var type width id name $end
            vtype = parts[1]
            width = int(parts[2])
            vid = parts[3]
            name = parts[4]
            signals[name] = {"id": vid, "width": width, "type": vtype}
    return signals


def _parse_vcd_changes(vcd: str) -> dict[int, dict[str, int]]:
    """Parse VCD value changes into {tick: {id: value}}.

    Skips the $dumpvars section.
    """
    changes: dict[int, dict[str, int]] = {}
    current_tick = None
    in_dumpvars = False

    for line in vcd.splitlines():
        line = line.strip()
        if line == "$dumpvars":
            in_dumpvars = True
            continue
        if in_dumpvars:
            if line == "$end":
                in_dumpvars = False
            continue
        if line.startswith("#"):
            current_tick = int(line[1:])
            if current_tick not in changes:
                changes[current_tick] = {}
        elif current_tick is not None and line:
            if line.startswith("b"):
                # Multi-bit: "bXXXX id"
                parts = line.split()
                bits = parts[0][1:]  # strip 'b'
                vid = parts[1]
                changes[current_tick][vid] = int(bits, 2)
            elif len(line) >= 2 and line[0] in "01xzXZ":
                # 1-bit: "0id" or "1id"
                val = 1 if line[0] == "1" else 0
                vid = line[1:]
                changes[current_tick][vid] = val

    return changes


def _parse_vcd_dumpvars(vcd: str) -> dict[str, int]:
    """Parse the $dumpvars section into {id: value}."""
    result: dict[str, int] = {}
    in_dumpvars = False

    for line in vcd.splitlines():
        line = line.strip()
        if line == "$dumpvars":
            in_dumpvars = True
            continue
        if not in_dumpvars:
            continue
        if line == "$end":
            break
        if line.startswith("b"):
            parts = line.split()
            bits = parts[0][1:]
            vid = parts[1]
            result[vid] = int(bits, 2)
        elif len(line) >= 2 and line[0] in "01xzXZ":
            val = 1 if line[0] == "1" else 0
            vid = line[1:]
            result[vid] = val

    return result


class TestMakeId:
    def test_first_ids(self):
        ids = [_make_id(i) for i in range(5)]
        assert len(set(ids)) == 5  # all unique

    def test_deterministic(self):
        assert _make_id(0) == _make_id(0)
        assert _make_id(42) == _make_id(42)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _make_id(-1)


class TestParseTimescale:
    def test_1ns(self):
        assert _parse_timescale_ns("1ns") == 1

    def test_10ns(self):
        assert _parse_timescale_ns("10ns") == 10

    def test_1us(self):
        assert _parse_timescale_ns("1us") == 1000

    def test_1ps(self):
        assert _parse_timescale_ns("1ps") == 1  # min 1

    def test_1ms(self):
        assert _parse_timescale_ns("1ms") == 1_000_000


class TestFormatValue:
    def test_1bit_zero(self):
        assert _format_value(0, 1) == "0"

    def test_1bit_one(self):
        assert _format_value(1, 1) == "1"

    def test_8bit(self):
        result = _format_value(0xAB, 8)
        assert result == "b10101011"

    def test_8bit_zero(self):
        result = _format_value(0, 8)
        assert result == "b00000000"

    def test_4bit(self):
        result = _format_value(0xF, 4)
        assert result == "b1111"

    def test_negative_signed(self):
        # -1 in 8-bit two's complement = 0xFF = 11111111
        result = _format_value(-1, 8)
        assert result == "b11111111"


class TestVCDStructure:
    """Test VCD header, signal declarations, and overall structure."""

    def _make_simple_module(self) -> Module:
        """A simple combinational module: y = a + b."""
        return Module(
            name="adder",
            ports=(
                Port(Signal("a", Shape(8)), PortDirection.INPUT),
                Port(Signal("b", Shape(8)), PortDirection.INPUT),
                Port(Signal("y", Shape(8)), PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", Binary(Shape(9), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
        )

    def test_header_sections(self):
        mod = self._make_simple_module()
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 1, "b": 2})
        vcd = cm.traces_to_vcd(traces)

        assert "$version" in vcd
        assert "dau-sim" in vcd
        assert "$timescale" in vcd
        assert "$scope module adder $end" in vcd
        assert "$enddefinitions $end" in vcd
        assert "$dumpvars" in vcd

    def test_signal_declarations(self):
        mod = self._make_simple_module()
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 1, "b": 2})
        vcd = cm.traces_to_vcd(traces)

        sigs = _parse_vcd_signals(vcd)
        assert "a" in sigs
        assert "b" in sigs
        assert "y" in sigs
        assert sigs["a"]["width"] == 8
        assert sigs["y"]["width"] == 8
        assert sigs["a"]["type"] == "wire"

    def test_custom_timescale(self):
        mod = self._make_simple_module()
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 1, "b": 2})
        vcd = cm.traces_to_vcd(traces, timescale="10ns")
        assert "$timescale 10ns $end" in vcd

    def test_1bit_signals(self):
        """1-bit signals should be declared with width 1."""
        mod = Module(
            name="buf",
            ports=(
                Port(Signal("a", Shape(1)), PortDirection.INPUT),
                Port(Signal("y", Shape(1)), PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", SignalRef(Shape(1), "a")),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 1})
        vcd = cm.traces_to_vcd(traces)
        sigs = _parse_vcd_signals(vcd)
        assert sigs["a"]["width"] == 1
        assert sigs["y"]["width"] == 1


class TestVCDEndToEnd:
    """Simulate designs and verify VCD waveform content."""

    def test_combinational_adder(self):
        """Comb adder: a=10, b=20 → y=30 for all time."""
        mod = Module(
            name="adder",
            ports=(
                Port(Signal("a", Shape(8)), PortDirection.INPUT),
                Port(Signal("b", Shape(8)), PortDirection.INPUT),
                Port(Signal("y", Shape(8)), PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", Binary(Shape(9), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=3, inputs={"a": 10, "b": 20})
        vcd = cm.traces_to_vcd(traces)

        sigs = _parse_vcd_signals(vcd)
        dumpvars = _parse_vcd_dumpvars(vcd)

        # Initial y value should be 30
        y_id = sigs["y"]["id"]
        assert dumpvars[y_id] == 30

    def test_sequential_counter(self):
        """Counter: count goes 1, 2, 3, ..."""
        mod = Module(
            name="counter",
            ports=(
                Port(Signal("clk", Shape(1)), PortDirection.INPUT),
                Port(Signal("rst", Shape(1)), PortDirection.INPUT),
                Port(Signal("count", Shape(8)), PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst", rst_style=ResetStyle.SYNC),),
            seq_blocks=(
                SeqBlock("sync", stmts=(Assign("count", Binary(Shape(9), BinaryOp.ADD, SignalRef(Shape(8), "count"), Const(Shape(8), 1))),)),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=5, inputs={})
        vcd = cm.traces_to_vcd(traces)

        sigs = _parse_vcd_signals(vcd)
        dumpvars = _parse_vcd_dumpvars(vcd)
        changes = _parse_vcd_changes(vcd)

        count_id = sigs["count"]["id"]

        # Initial value should be 0 (pre-first-edge reset state)
        assert dumpvars[count_id] == 0

        # Changes should include all posedge values
        change_vals = []
        for tick in sorted(changes.keys()):
            if count_id in changes[tick]:
                change_vals.append(changes[tick][count_id])
        assert change_vals == [1, 2, 3, 4, 5]

    def test_timestamp_spacing(self):
        """Consecutive timestamps should be evenly spaced."""
        mod = Module(
            name="counter",
            ports=(
                Port(Signal("clk", Shape(1)), PortDirection.INPUT),
                Port(Signal("rst", Shape(1)), PortDirection.INPUT),
                Port(Signal("count", Shape(8)), PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst", rst_style=ResetStyle.SYNC),),
            seq_blocks=(
                SeqBlock("sync", stmts=(Assign("count", Binary(Shape(9), BinaryOp.ADD, SignalRef(Shape(8), "count"), Const(Shape(8), 1))),)),
            ),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=5, inputs={})
        vcd = cm.traces_to_vcd(traces)
        changes = _parse_vcd_changes(vcd)

        ticks = sorted(changes.keys())
        assert len(ticks) >= 2
        # All intervals should be equal
        intervals = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
        assert len(set(intervals)) == 1

    def test_no_module_fallback(self):
        """Without module metadata, signals default to 1-bit wire."""
        t0 = datetime(2000, 1, 1)
        traces = {
            "x": [(t0, 0), (t0 + timedelta(microseconds=1), 1)],
        }
        vcd = traces_to_vcd(traces)
        sigs = _parse_vcd_signals(vcd)
        assert sigs["x"]["width"] == 1
        assert sigs["x"]["type"] == "wire"
        assert "$scope module top $end" in vcd

    def test_custom_scope(self):
        t0 = datetime(2000, 1, 1)
        traces = {"x": [(t0, 0)]}
        vcd = traces_to_vcd(traces, scope="my_dut")
        assert "$scope module my_dut $end" in vcd


class TestVCDFileIO:
    """Test writing VCD to files."""

    def test_write_vcd_file(self):
        """write_vcd should create a valid VCD file."""
        mod = Module(
            name="test_mod",
            ports=(
                Port(Signal("a", Shape(4)), PortDirection.INPUT),
                Port(Signal("y", Shape(4)), PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign("y", SignalRef(Shape(4), "a")),)),),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=2, inputs={"a": 5})

        with tempfile.NamedTemporaryFile(suffix=".vcd", delete=False) as f:
            path = f.name

        try:
            cm.write_vcd(path, traces)
            assert os.path.exists(path)
            contents = open(path).read()
            assert "$version" in contents
            assert "$var" in contents
            assert "$dumpvars" in contents
        finally:
            os.unlink(path)

    def test_standalone_write_vcd(self):
        """write_vcd standalone function should work."""
        t0 = datetime(2000, 1, 1)
        traces = {"sig": [(t0, 1), (t0 + timedelta(microseconds=1), 0)]}

        with tempfile.NamedTemporaryFile(suffix=".vcd", delete=False) as f:
            path = f.name

        try:
            write_vcd(path, traces)
            contents = open(path).read()
            assert "$dumpvars" in contents
        finally:
            os.unlink(path)


class TestAmaranthVCD:
    """Full pipeline: Amaranth design → simulate → VCD."""

    def test_amaranth_counter_vcd(self):
        from amaranth.hdl import Module as AModule
        from amaranth.lib.wiring import Component, Out

        from dau_sim.frontends import from_amaranth

        class Counter(Component):
            count: Out(8)

            def elaborate(self, platform):
                m = AModule()
                m.d.sync += self.count.eq(self.count + 1)
                return m

        ir_mod = from_amaranth(Counter())
        cm = compile_module(ir_mod)
        traces = cm.run(cycles=10, inputs={})
        vcd = cm.traces_to_vcd(traces)

        sigs = _parse_vcd_signals(vcd)
        assert "count" in sigs
        assert sigs["count"]["width"] == 8

        dumpvars = _parse_vcd_dumpvars(vcd)
        count_id = sigs["count"]["id"]
        assert dumpvars[count_id] == 0  # pre-first-edge reset state

        changes = _parse_vcd_changes(vcd)
        count_changes = []
        for tick in sorted(changes.keys()):
            if count_id in changes[tick]:
                count_changes.append(changes[tick][count_id])
        assert count_changes == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_amaranth_adder_vcd(self):
        from amaranth.hdl import Module as AModule
        from amaranth.lib.wiring import Component, In, Out

        from dau_sim.frontends import from_amaranth

        class Adder(Component):
            a: In(8)
            b: In(8)
            y: Out(8)

            def elaborate(self, platform):
                m = AModule()
                m.d.comb += self.y.eq(self.a + self.b)
                return m

        ir_mod = from_amaranth(Adder())
        cm = compile_module(ir_mod)
        traces = cm.run(cycles=1, inputs={"a": 100, "b": 55})
        vcd = cm.traces_to_vcd(traces)

        sigs = _parse_vcd_signals(vcd)
        dumpvars = _parse_vcd_dumpvars(vcd)

        a_id = sigs["a"]["id"]
        b_id = sigs["b"]["id"]
        y_id = sigs["y"]["id"]
        assert dumpvars[a_id] == 100
        assert dumpvars[b_id] == 55
        assert dumpvars[y_id] == 155

    def test_amaranth_counter_vcd_file(self):
        """Full pipeline to VCD file."""
        from amaranth.hdl import Module as AModule
        from amaranth.lib.wiring import Component, Out

        from dau_sim.frontends import from_amaranth

        class Counter(Component):
            count: Out(4)

            def elaborate(self, platform):
                m = AModule()
                m.d.sync += self.count.eq(self.count + 1)
                return m

        ir_mod = from_amaranth(Counter())
        cm = compile_module(ir_mod)
        traces = cm.run(cycles=20, inputs={})

        with tempfile.NamedTemporaryFile(suffix=".vcd", delete=False) as f:
            path = f.name

        try:
            cm.write_vcd(path, traces)
            contents = open(path).read()
            sigs = _parse_vcd_signals(contents)
            assert "count" in sigs
            assert sigs["count"]["width"] == 4

            # Verify counter wraps at 16
            dumpvars = _parse_vcd_dumpvars(contents)
            changes = _parse_vcd_changes(contents)
            count_id = sigs["count"]["id"]

            all_vals = [dumpvars[count_id]]
            for tick in sorted(changes.keys()):
                if count_id in changes[tick]:
                    all_vals.append(changes[tick][count_id])
            # Should wrap: 0,1,2,...,15,0,1,2,3,4
            assert all_vals[15] == 15
            assert all_vals[16] == 0
            assert all_vals[17] == 1
        finally:
            os.unlink(path)
