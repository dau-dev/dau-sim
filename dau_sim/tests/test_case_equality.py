"""``===`` and ``!==`` compare X and Z bits as values and never yield X;
``==`` and ``!=`` yield X when an operand has an unknown bit. On two-state
values the two pairs agree."""

from dau_sim.compiler import compile_module
from dau_sim.compiler.eval4 import eval_expr_4
from dau_sim.frontends import parse_sv
from dau_sim.ir import Binary, BinaryOp, Const, Shape, SignalRef
from dau_sim.ir.printer import fmt_expr
from dau_sim.ir.types import FourState

B = Shape(1)
W = Shape(4)


def _cmp(op, left, right):
    return eval_expr_4(Binary(shape=B, op=op, left=SignalRef(shape=W, name="l"), right=SignalRef(shape=W, name="r")), {"l": left, "r": right})


def test_case_equality_treats_x_and_z_as_values():
    x, z, five = FourState.x(W), FourState.z(W), FourState.from_int(5, W)
    assert _cmp(BinaryOp.CASE_EQ, x, x).to_int == 1
    assert _cmp(BinaryOp.CASE_EQ, z, z).to_int == 1
    assert _cmp(BinaryOp.CASE_EQ, x, z).to_int == 0
    assert _cmp(BinaryOp.CASE_EQ, x, five).to_int == 0
    assert _cmp(BinaryOp.CASE_NE, x, five).to_int == 1
    assert _cmp(BinaryOp.CASE_NE, x, x).to_int == 0
    # partial unknowns: 4'b01x0 === 4'b01x0 is 1, against 4'b0100 it is 0
    partial = FourState(shape=W, aval=0b0110, bval=0b0010)
    assert _cmp(BinaryOp.CASE_EQ, partial, partial).to_int == 1
    assert _cmp(BinaryOp.CASE_EQ, partial, FourState.from_int(0b0100, W)).to_int == 0


def test_logical_equality_is_unknown_when_an_operand_is():
    x, five = FourState.x(W), FourState.from_int(5, W)
    assert _cmp(BinaryOp.EQ, x, x).to_int is None
    assert _cmp(BinaryOp.NE, x, five).to_int is None
    assert _cmp(BinaryOp.EQ, five, five).to_int == 1


def test_the_pairs_agree_on_two_state_values():
    five, six = FourState.from_int(5, W), FourState.from_int(6, W)
    assert _cmp(BinaryOp.CASE_EQ, five, five).to_int == _cmp(BinaryOp.EQ, five, five).to_int == 1
    assert _cmp(BinaryOp.CASE_NE, five, six).to_int == _cmp(BinaryOp.NE, five, six).to_int == 1


def test_systemverilog_case_equality_lowers_to_its_own_operators():
    mod = parse_sv("""
        module ceq(input wire [3:0] a, input wire [3:0] b, output wire y, output wire n, output wire e);
            assign y = (a === b);
            assign n = (a !== b);
            assign e = (a == b);
        endmodule
    """)
    ops = {stmt.target: stmt.value.op for cb in mod.comb_blocks for stmt in cb.stmts}
    assert (ops["y"], ops["n"], ops["e"]) == (BinaryOp.CASE_EQ, BinaryOp.CASE_NE, BinaryOp.EQ)
    assert fmt_expr(Binary(shape=B, op=BinaryOp.CASE_EQ, left=Const(shape=W, value=1), right=Const(shape=W, value=1))).count("===") == 1
    # two-state execution: both pairs are plain equality
    traces = compile_module(mod).run(cycles=1, inputs={"a": 7, "b": 7})
    assert [v for _, v in traces["y"]] == [1] and [v for _, v in traces["n"]] == [0] and [v for _, v in traces["e"]] == [1]
