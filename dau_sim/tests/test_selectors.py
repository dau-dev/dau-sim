"""Tests for signal selectors (glob/regex filtering of traces)."""

import unittest
from datetime import datetime

from dau_sim.adapters.selectors import match_signals, select_signals

SAMPLE_NAMES = ["clk", "rst", "en", "count", "data_0", "data_1", "data_2", "internal_reg"]
SAMPLE_TRACES = {name: [(datetime(2000, 1, 1), i)] for i, name in enumerate(SAMPLE_NAMES)}


class TestMatchSignals(unittest.TestCase):
    def test_no_filters(self):
        result = match_signals(SAMPLE_NAMES)
        self.assertEqual(result, list(SAMPLE_NAMES))

    def test_include_exact(self):
        result = match_signals(SAMPLE_NAMES, include=["clk", "rst"])
        self.assertEqual(result, ["clk", "rst"])

    def test_include_glob_star(self):
        result = match_signals(SAMPLE_NAMES, include=["data_*"])
        self.assertEqual(result, ["data_0", "data_1", "data_2"])

    def test_include_glob_question(self):
        result = match_signals(SAMPLE_NAMES, include=["data_?"])
        self.assertEqual(result, ["data_0", "data_1", "data_2"])

    def test_exclude_exact(self):
        result = match_signals(SAMPLE_NAMES, exclude=["clk", "rst"])
        self.assertNotIn("clk", result)
        self.assertNotIn("rst", result)
        self.assertIn("en", result)

    def test_exclude_glob(self):
        result = match_signals(SAMPLE_NAMES, exclude=["*internal*"])
        self.assertNotIn("internal_reg", result)
        self.assertIn("clk", result)

    def test_include_and_exclude(self):
        result = match_signals(
            SAMPLE_NAMES,
            include=["data_*", "clk"],
            exclude=["data_2"],
        )
        self.assertEqual(set(result), {"clk", "data_0", "data_1"})
        self.assertEqual(len(result), 3)

    def test_regex_include(self):
        result = match_signals(
            SAMPLE_NAMES,
            include=[r"data_\d+"],
            regex=True,
        )
        self.assertEqual(result, ["data_0", "data_1", "data_2"])

    def test_regex_exclude(self):
        result = match_signals(
            SAMPLE_NAMES,
            exclude=[r"^(clk|rst)$"],
            regex=True,
        )
        self.assertNotIn("clk", result)
        self.assertNotIn("rst", result)
        self.assertIn("en", result)

    def test_no_match_returns_empty(self):
        result = match_signals(SAMPLE_NAMES, include=["nonexistent"])
        self.assertEqual(result, [])

    def test_preserves_order(self):
        result = match_signals(SAMPLE_NAMES, include=["count", "en", "clk"])
        # Order follows pattern order (first match per pattern)
        self.assertEqual(result, ["count", "en", "clk"])

    def test_empty_names(self):
        result = match_signals([], include=["*"])
        self.assertEqual(result, [])


class TestSelectSignals(unittest.TestCase):
    def test_include_filters_traces(self):
        result = select_signals(SAMPLE_TRACES, include=["clk", "en"])
        self.assertEqual(set(result.keys()), {"clk", "en"})
        self.assertEqual(result["clk"], SAMPLE_TRACES["clk"])

    def test_exclude_filters_traces(self):
        result = select_signals(SAMPLE_TRACES, exclude=["*internal*"])
        self.assertNotIn("internal_reg", result)
        self.assertIn("clk", result)

    def test_glob_wildcard(self):
        result = select_signals(SAMPLE_TRACES, include=["data_*"])
        self.assertEqual(set(result.keys()), {"data_0", "data_1", "data_2"})

    def test_no_filter_returns_all(self):
        result = select_signals(SAMPLE_TRACES)
        self.assertEqual(set(result.keys()), set(SAMPLE_TRACES.keys()))


class TestIntegration(unittest.TestCase):
    def test_vcd_with_signal_filter(self):
        """Test that CompiledModule.traces_to_vcd supports signals param."""
        from dau_sim.compiler import compile_module
        from dau_sim.ir import (
            Assign,
            Binary,
            BinaryOp,
            CombBlock,
            Module,
            Port,
            PortDirection,
            Shape,
            Signal,
            SignalRef,
        )

        a = Signal("a", Shape(8))
        b = Signal("b", Shape(8))
        y = Signal("y", Shape(8))
        mod = Module(
            name="adder",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(b, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            signals=(),
            clock_domains=(),
            comb_blocks=(CombBlock(stmts=(Assign("y", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
            seq_blocks=(),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 10, "b": 20})

        # Only include 'y' in VCD
        vcd = cm.traces_to_vcd(traces, signals=["y"])
        self.assertIn("y", vcd)
        # 'a' and 'b' should not appear as signal declarations
        lines = vcd.split("\n")
        var_lines = [line for line in lines if line.strip().startswith("$var")]
        signal_names = [line.split()[-2] for line in var_lines]
        self.assertEqual(signal_names, ["y"])

    def test_vcd_with_exclude(self):
        """Test exclude parameter on traces_to_vcd."""
        from dau_sim.compiler import compile_module
        from dau_sim.ir import (
            Assign,
            Binary,
            BinaryOp,
            CombBlock,
            Module,
            Port,
            PortDirection,
            Shape,
            Signal,
            SignalRef,
        )

        a = Signal("a", Shape(8))
        b = Signal("b", Shape(8))
        y = Signal("y", Shape(8))
        mod = Module(
            name="adder",
            ports=(
                Port(a, PortDirection.INPUT),
                Port(b, PortDirection.INPUT),
                Port(y, PortDirection.OUTPUT),
            ),
            signals=(),
            clock_domains=(),
            comb_blocks=(CombBlock(stmts=(Assign("y", Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))),)),),
            seq_blocks=(),
        )
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 10, "b": 20})

        # Exclude inputs, keep only output
        vcd = cm.traces_to_vcd(traces, exclude=["a", "b"])
        lines = vcd.split("\n")
        var_lines = [line for line in lines if line.strip().startswith("$var")]
        signal_names = [line.split()[-2] for line in var_lines]
        self.assertEqual(signal_names, ["y"])


if __name__ == "__main__":
    unittest.main()
