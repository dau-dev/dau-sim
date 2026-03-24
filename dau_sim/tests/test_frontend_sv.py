from datetime import timedelta

import pytest

from dau_sim.compiler import compile_module
from dau_sim.frontends import parse_sv
from dau_sim.ir import (
    EdgePolarity,
    PortDirection,
    ResetStyle,
    Shape,
)


class TestModuleExtraction:
    """Test port/signal/domain extraction from SV source."""

    def test_basic_ports(self):
        mod = parse_sv("""
            module basic(input wire a, output wire b);
                assign b = a;
            endmodule
        """)
        assert mod.name == "basic"
        assert len(mod.ports) == 2
        assert mod.ports[0].name == "a"
        assert mod.ports[0].direction == PortDirection.INPUT
        assert mod.ports[0].shape == Shape(1, False)
        assert mod.ports[1].name == "b"
        assert mod.ports[1].direction == PortDirection.OUTPUT

    def test_wide_ports(self):
        mod = parse_sv("""
            module wide(input wire [7:0] data_in, output wire [15:0] data_out);
                assign data_out = {8'd0, data_in};
            endmodule
        """)
        assert mod.ports[0].shape == Shape(8, False)
        assert mod.ports[1].shape == Shape(16, False)

    def test_signed_ports(self):
        mod = parse_sv("""
            module signed_test(input wire signed [7:0] a, output wire signed [7:0] b);
                assign b = a;
            endmodule
        """)
        assert mod.ports[0].shape == Shape(8, True)
        assert mod.ports[1].shape == Shape(8, True)

    def test_multiple_ports(self):
        mod = parse_sv("""
            module multi(
                input wire clk,
                input wire rst,
                input wire [3:0] a,
                input wire [3:0] b,
                output wire [3:0] sum,
                output wire carry
            );
                assign {carry, sum} = a + b;
            endmodule
        """)
        assert len(mod.ports) == 6
        input_ports = [p for p in mod.ports if p.direction == PortDirection.INPUT]
        output_ports = [p for p in mod.ports if p.direction == PortDirection.OUTPUT]
        assert len(input_ports) == 4
        assert len(output_ports) == 2

    def test_top_module_selection(self):
        mod = parse_sv(
            """
            module inner(input wire a, output wire b);
                assign b = a;
            endmodule
            module outer(input wire x, output wire y);
                inner inst(.a(x), .b(y));
            endmodule
        """,
            top="outer",
        )
        assert mod.name == "outer"

    def test_top_module_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            parse_sv(
                """
                module foo(input wire a, output wire b);
                    assign b = a;
                endmodule
            """,
                top="bar",
            )


class TestExpressionLowering:
    """Test that SV expressions lower to correct IR expression trees."""

    def test_constant_assign(self):
        mod = parse_sv("""
            module const_test(output wire [3:0] y);
                assign y = 4'd5;
            endmodule
        """)
        assert len(mod.comb_blocks) == 1
        stmt = mod.comb_blocks[0].stmts[0]
        assert stmt.target == "y"

    def test_binary_add(self):
        mod = parse_sv("""
            module add_test(input wire [3:0] a, b, output wire [3:0] y);
                assign y = a + b;
            endmodule
        """)
        stmt = mod.comb_blocks[0].stmts[0]
        assert stmt.target == "y"

    def test_bitwise_ops(self):
        mod = parse_sv("""
            module bitwise(
                input wire [3:0] a, b,
                output wire [3:0] y_and, y_or, y_xor
            );
                assign y_and = a & b;
                assign y_or = a | b;
                assign y_xor = a ^ b;
            endmodule
        """)
        assert len(mod.comb_blocks) == 3

    def test_unary_not(self):
        mod = parse_sv("""
            module not_test(input wire [3:0] a, output wire [3:0] y);
                assign y = ~a;
            endmodule
        """)
        assert len(mod.comb_blocks) == 1

    def test_ternary_mux(self):
        mod = parse_sv("""
            module mux_test(
                input wire sel,
                input wire [3:0] a, b,
                output wire [3:0] y
            );
                assign y = sel ? a : b;
            endmodule
        """)
        assert len(mod.comb_blocks) == 1

    def test_concatenation(self):
        mod = parse_sv("""
            module cat_test(
                input wire [3:0] a, b,
                output wire [7:0] y
            );
                assign y = {a, b};
            endmodule
        """)
        assert len(mod.comb_blocks) == 1
        stmt = mod.comb_blocks[0].stmts[0]
        assert stmt.target == "y"

    def test_bit_slice(self):
        mod = parse_sv("""
            module slice_test(input wire [7:0] a, output wire [3:0] y);
                assign y = a[7:4];
            endmodule
        """)
        assert len(mod.comb_blocks) == 1

    def test_single_bit_select(self):
        mod = parse_sv("""
            module bit_sel(input wire [7:0] a, output wire y);
                assign y = a[0];
            endmodule
        """)
        assert len(mod.comb_blocks) == 1

    def test_comparison_ops(self):
        mod = parse_sv("""
            module cmp(input wire [3:0] a, b, output wire eq, lt, gt);
                assign eq = (a == b);
                assign lt = (a < b);
                assign gt = (a > b);
            endmodule
        """)
        assert len(mod.comb_blocks) == 3

    def test_shift_ops(self):
        mod = parse_sv("""
            module shift(input wire [7:0] a, output wire [7:0] left, right);
                assign left = a << 2;
                assign right = a >> 2;
            endmodule
        """)
        assert len(mod.comb_blocks) == 2


# ═══════════════════════════════════════════════════════════════════
# Statement lowering
# ═══════════════════════════════════════════════════════════════════


class TestStatementLowering:
    """Test SV always blocks lower to correct IR blocks."""

    def test_always_comb_single(self):
        mod = parse_sv("""
            module comb_single(input wire [3:0] a, output reg [3:0] y);
                always_comb y = a;
            endmodule
        """)
        assert len(mod.comb_blocks) == 1
        assert len(mod.seq_blocks) == 0

    def test_always_comb_block(self):
        mod = parse_sv("""
            module comb_block(
                input wire [3:0] a, b,
                output reg [3:0] y, z
            );
                always_comb begin
                    y = a;
                    z = b;
                end
            endmodule
        """)
        assert len(mod.comb_blocks) == 1
        assert len(mod.comb_blocks[0].stmts) == 2

    def test_always_ff_posedge(self):
        mod = parse_sv("""
            module ff_pos(input wire clk, input wire [3:0] d, output reg [3:0] q);
                always_ff @(posedge clk) q <= d;
            endmodule
        """)
        assert len(mod.seq_blocks) == 1
        assert mod.seq_blocks[0].domain == "clk"
        assert mod.clock_domains[0].edge == EdgePolarity.POSEDGE

    def test_always_ff_negedge(self):
        mod = parse_sv("""
            module ff_neg(input wire clk, input wire [3:0] d, output reg [3:0] q);
                always_ff @(negedge clk) q <= d;
            endmodule
        """)
        assert mod.clock_domains[0].edge == EdgePolarity.NEGEDGE

    def test_if_else(self):
        mod = parse_sv("""
            module if_test(input wire clk, input wire sel, input wire [3:0] a, b, output reg [3:0] y);
                always_ff @(posedge clk) begin
                    if (sel) begin
                        y <= a;
                    end else begin
                        y <= b;
                    end
                end
            endmodule
        """)
        assert len(mod.seq_blocks) == 1
        stmts = mod.seq_blocks[0].stmts
        assert len(stmts) == 1  # single IfElse


# ═══════════════════════════════════════════════════════════════════
# Clock domain inference
# ═══════════════════════════════════════════════════════════════════


class TestClockDomainInference:
    """Test clock domain extraction from always blocks."""

    def test_single_clock_posedge(self):
        mod = parse_sv("""
            module clk_test(input wire clk, output reg q);
                always @(posedge clk) q <= 1'b1;
            endmodule
        """)
        assert len(mod.clock_domains) == 1
        d = mod.clock_domains[0]
        assert d.name == "clk"
        assert d.clk == "clk"
        assert d.edge == EdgePolarity.POSEDGE
        assert d.rst is None

    def test_async_reset_negedge(self):
        mod = parse_sv("""
            module async_rst(input wire clk, input wire rst_n, output reg q);
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        q <= 1'b0;
                    end else begin
                        q <= 1'b1;
                    end
                end
            endmodule
        """)
        d = mod.clock_domains[0]
        assert d.rst == "rst_n"
        assert d.rst_style == ResetStyle.ASYNC
        assert d.rst_active_high is False  # negedge → active-low

    def test_async_reset_posedge(self):
        mod = parse_sv("""
            module async_rst_hi(input wire clk, input wire rst, output reg q);
                always_ff @(posedge clk or posedge rst) begin
                    if (rst) begin
                        q <= 1'b0;
                    end else begin
                        q <= 1'b1;
                    end
                end
            endmodule
        """)
        d = mod.clock_domains[0]
        assert d.rst == "rst"
        assert d.rst_style == ResetStyle.ASYNC
        assert d.rst_active_high is True

    def test_no_duplicate_domains(self):
        mod = parse_sv("""
            module two_regs(input wire clk, input wire [3:0] a, b, output reg [3:0] qa, qb);
                always_ff @(posedge clk) qa <= a;
                always_ff @(posedge clk) qb <= b;
            endmodule
        """)
        assert len(mod.clock_domains) == 1
        assert len(mod.seq_blocks) == 2


# ═══════════════════════════════════════════════════════════════════
# End-to-end simulation: SV → IR → compile → simulate
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndSimulation:
    """Parse SV source, compile to CSP, simulate, and verify traces."""

    def test_combinational_adder(self):
        """4-bit ripple adder via continuous assign."""
        mod = parse_sv("""
            module adder(input wire [3:0] a, b, output wire [3:0] sum);
                assign sum = a + b;
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 3, "b": 5})
        vals = [v for _, v in traces["sum"]]
        assert vals[-1] == 8

    def test_combinational_mux(self):
        """2:1 mux via ternary."""
        mod = parse_sv("""
            module mux2(input wire sel, input wire [3:0] a, b, output wire [3:0] y);
                assign y = sel ? a : b;
            endmodule
        """)
        cm = compile_module(mod)
        # sel=1 → y=a=7
        traces = cm.run(cycles=1, inputs={"sel": 1, "a": 7, "b": 3})
        assert [v for _, v in traces["y"]][-1] == 7
        # sel=0 → y=b=3
        traces = cm.run(cycles=1, inputs={"sel": 0, "a": 7, "b": 3})
        assert [v for _, v in traces["y"]][-1] == 3

    def test_combinational_bitwise(self):
        """Bitwise AND, OR, XOR."""
        mod = parse_sv("""
            module bitwise(
                input wire [3:0] a, b,
                output wire [3:0] y_and, y_or, y_xor
            );
                assign y_and = a & b;
                assign y_or = a | b;
                assign y_xor = a ^ b;
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 0b1100, "b": 0b1010})
        assert [v for _, v in traces["y_and"]][-1] == 0b1000
        assert [v for _, v in traces["y_or"]][-1] == 0b1110
        assert [v for _, v in traces["y_xor"]][-1] == 0b0110

    def test_combinational_invert(self):
        """Bitwise NOT."""
        mod = parse_sv("""
            module inv(input wire [3:0] a, output wire [3:0] y);
                assign y = ~a;
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 0b0101})
        assert [v for _, v in traces["y"]][-1] == 0b1010

    def test_combinational_concat(self):
        """Concatenation {a, b}."""
        mod = parse_sv("""
            module cat(input wire [3:0] a, b, output wire [7:0] y);
                assign y = {a, b};
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 0xA, "b": 0x5})
        assert [v for _, v in traces["y"]][-1] == 0xA5

    def test_combinational_slice(self):
        """Bit slice a[7:4]."""
        mod = parse_sv("""
            module slic(input wire [7:0] a, output wire [3:0] y);
                assign y = a[7:4];
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 0xAB})
        assert [v for _, v in traces["y"]][-1] == 0xA

    def test_sequential_counter(self):
        """4-bit counter via always @(posedge clk)."""
        mod = parse_sv("""
            module counter(input wire clk, input wire rst, output reg [3:0] count);
                always @(posedge clk) begin
                    if (rst) begin
                        count <= 4'd0;
                    end else begin
                        count <= count + 4'd1;
                    end
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=10,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"rst": 0},
        )
        count_vals = [v for _, v in traces["count"]]
        # Should count 1..10 (starts from init=0, first edge increments to 1)
        assert count_vals == list(range(1, 11))

    def test_sequential_counter_with_reset(self):
        """Counter held in reset: should stay at 0."""
        mod = parse_sv("""
            module counter(input wire clk, input wire rst, output reg [3:0] count);
                always @(posedge clk) begin
                    if (rst) begin
                        count <= 4'd0;
                    end else begin
                        count <= count + 4'd1;
                    end
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=5,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"rst": 1},
        )
        count_vals = [v for _, v in traces["count"]]
        assert all(v == 0 for v in count_vals)

    def test_sequential_dff(self):
        """D flip-flop: q follows d on posedge clk."""
        mod = parse_sv("""
            module dff(input wire clk, input wire [7:0] d, output reg [7:0] q);
                always_ff @(posedge clk) q <= d;
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=3,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"d": 42},
        )
        q_vals = [v for _, v in traces["q"]]
        # After first posedge, q = d = 42
        assert q_vals[0] == 42

    def test_sequential_async_reset(self):
        """always_ff with async reset (negedge rst_n)."""
        mod = parse_sv("""
            module async_dff(
                input wire clk,
                input wire rst_n,
                input wire [3:0] d,
                output reg [3:0] q
            );
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        q <= 4'd0;
                    end else begin
                        q <= d;
                    end
                end
            endmodule
        """)
        cm = compile_module(mod)
        # rst_n=1 (not reset) → q should follow d
        traces = cm.run(
            cycles=3,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"rst_n": 1, "d": 9},
        )
        q_vals = [v for _, v in traces["q"]]
        assert q_vals[0] == 9

    def test_sequential_shift_register(self):
        """4-bit shift register: shifts left on each posedge, bottom bit = serial_in."""
        mod = parse_sv("""
            module shift_reg(
                input wire clk,
                input wire serial_in,
                output reg [3:0] q
            );
                always_ff @(posedge clk) begin
                    q <= {q[2:0], serial_in};
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=4,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"serial_in": 1},
        )
        q_vals = [v for _, v in traces["q"]]
        # serial_in=1 shifts in: 0b0001, 0b0011, 0b0111, 0b1111
        assert q_vals == [1, 3, 7, 15]

    def test_sequential_counter_100_cycles(self):
        """Graduation test: 8-bit counter over 100 cycles."""
        mod = parse_sv("""
            module counter100(input wire clk, output reg [7:0] count);
                always @(posedge clk) begin
                    count <= count + 8'd1;
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=100,
            clocks={"clk": timedelta(microseconds=1)},
        )
        count_vals = [v for _, v in traces["count"]]
        assert count_vals == list(range(1, 101))

    def test_combined_comb_and_seq(self):
        """Module with both combinational and sequential logic."""
        mod = parse_sv("""
            module comb_seq(
                input wire clk,
                input wire [3:0] a,
                output reg [3:0] q,
                output wire [3:0] doubled
            );
                always_ff @(posedge clk) q <= a;
                assign doubled = q + q;
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=3,
            clocks={"clk": timedelta(microseconds=1)},
            inputs={"a": 5},
        )
        q_vals = [v for _, v in traces["q"]]
        doubled_vals = [v for _, v in traces["doubled"]]
        # After first posedge: q=5, doubled=10
        assert q_vals[0] == 5
        assert doubled_vals[-1] == 10

    def test_always_comb_simulation(self):
        """always_comb block in simulation."""
        mod = parse_sv("""
            module comb_sim(
                input wire [3:0] a, b,
                output reg [3:0] y, z
            );
                always_comb begin
                    y = a + b;
                    z = a - b;
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(cycles=1, inputs={"a": 10, "b": 3})
        assert [v for _, v in traces["y"]][-1] == 13
        assert [v for _, v in traces["z"]][-1] == 7

    def test_negedge_counter(self):
        """Counter on negedge clock."""
        mod = parse_sv("""
            module neg_ctr(input wire clk, output reg [3:0] count);
                always @(negedge clk) begin
                    count <= count + 4'd1;
                end
            endmodule
        """)
        cm = compile_module(mod)
        traces = cm.run(
            cycles=5,
            clocks={"clk": timedelta(microseconds=1)},
        )
        count_vals = [v for _, v in traces["count"]]
        assert count_vals == list(range(1, 6))


# ═══════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Test that errors are reported properly."""

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="[Cc]ompilation error"):
            parse_sv("module bad(input wire a, output wire  endmodule")

    def test_no_top_instances(self):
        with pytest.raises(ValueError, match="No top-level module"):
            parse_sv("")

    def test_wrong_top_name(self):
        with pytest.raises(ValueError, match="not found"):
            parse_sv(
                "module foo(input wire a, output wire b); assign b = a; endmodule",
                top="nonexistent",
            )


# ═══════════════════════════════════════════════════════════════════
# dau-build bridge
# ═══════════════════════════════════════════════════════════════════

_dau_build_available = False
try:
    from dau_build import Module as DauBuildModule

    _dau_build_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _dau_build_available, reason="dau-build not installed")
class TestDauBuildBridge:
    """Test the from_dau_build bridge between dau-build and dau-sim."""

    def test_bridge_from_str(self):
        """Round-trip: dau_build.Module.from_str → from_dau_build → IR Module."""
        from dau_sim.frontends import from_dau_build

        src = "module adder(input wire [7:0] a, input wire [7:0] b, output wire [7:0] y); assign y = a + b; endmodule"
        db_mod = DauBuildModule.from_str(src)
        ir_mod = from_dau_build(db_mod)
        assert ir_mod.name == "adder"
        port_names = {p.name for p in ir_mod.ports}
        assert port_names == {"a", "b", "y"}

    def test_bridge_from_file(self, tmp_path):
        """Bridge with source_path set via from_file."""
        from dau_sim.frontends import from_dau_build

        sv = tmp_path / "counter.sv"
        sv.write_text("""\
module counter(input wire clk, output reg [3:0] count);
    always @(posedge clk) count <= count + 4'd1;
endmodule
""")
        db_mod = DauBuildModule.from_file(sv)
        ir_mod = from_dau_build(db_mod)
        assert ir_mod.name == "counter"
        assert len(ir_mod.clock_domains) == 1
        assert ir_mod.clock_domains[0].clk == "clk"

    def test_bridge_simulation(self):
        """Full pipeline: dau_build.Module → dau-sim IR → compile → run."""
        from dau_sim.compiler.compile import compile_module
        from dau_sim.frontends import from_dau_build

        src = "module inv(input wire [3:0] a, output wire [3:0] y); assign y = ~a; endmodule"
        db_mod = DauBuildModule.from_str(src)
        ir_mod = from_dau_build(db_mod)
        cm = compile_module(ir_mod)
        traces = cm.run(cycles=1, inputs={"a": 5})
        y_vals = [v for _, v in traces["y"]]
        assert y_vals[-1] == (~5) & 0xF  # 10

    def test_bridge_no_source_raises(self):
        """Bridge raises if Module has no source_path and no node."""
        from dau_sim.frontends import from_dau_build

        db_mod = DauBuildModule(name="empty")
        with pytest.raises(ValueError, match="no source_path"):
            from_dau_build(db_mod)

    def test_bridge_port_consistency(self):
        """Ports from dau-build and dau-sim should agree on structure."""
        from dau_sim.frontends import from_dau_build

        src = "module m(input wire [15:0] x, output wire [7:0] y); assign y = x[7:0]; endmodule"
        db_mod = DauBuildModule.from_str(src)

        # dau-build structural info
        assert len(db_mod.inputs) == 1
        assert db_mod.inputs[0].name == "x"
        assert len(db_mod.outputs) == 1
        assert db_mod.outputs[0].name == "y"

        # dau-sim IR via bridge
        ir_mod = from_dau_build(db_mod)
        ir_ports = {p.name: p for p in ir_mod.ports}
        assert ir_ports["x"].shape.width == 16
        assert ir_ports["y"].shape.width == 8
