import pytest

from dau_sim.compiler import CombLoopError, compile_module
from dau_sim.compiler.depanalysis import (
    build_assignments,
    collect_reads,
    collect_stmt_writes,
    topological_sort,
)
from dau_sim.compiler.resolve import resolve_drivers
from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    FourState,
    Module,
    NetKind,
    Port,
    PortDirection,
    Shape,
    Signal,
    SignalRef,
)


class TestDependencyAnalysis:
    def test_collect_reads_simple(self):
        expr = Binary(
            shape=Shape(8),
            op=BinaryOp.ADD,
            left=SignalRef(shape=Shape(8), name="a"),
            right=SignalRef(shape=Shape(8), name="b"),
        )
        assert collect_reads(expr) == {"a", "b"}

    def test_collect_reads_const(self):
        assert collect_reads(Const(shape=Shape(8), value=5)) == set()

    def test_collect_writes(self):
        stmt = Assign(target="out", value=Const(shape=Shape(8), value=0))
        assert collect_stmt_writes(stmt) == {"out"}

    def test_topological_sort_chain(self):
        """a → b → c (three blocks in a chain)."""
        stmts = [
            (0, (Assign(target="c", value=SignalRef(shape=Shape(8), name="b")),)),
            (1, (Assign(target="b", value=SignalRef(shape=Shape(8), name="a")),)),
            (2, (Assign(target="a", value=Const(shape=Shape(8), value=0)),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)
        # a first, then b, then c
        names = [list(a.writes)[0] for a in ordered]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_topological_sort_parallel(self):
        """Independent blocks — any order is fine."""
        stmts = [
            (0, (Assign(target="x", value=Const(shape=Shape(8), value=1)),)),
            (1, (Assign(target="y", value=Const(shape=Shape(8), value=2)),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)
        assert len(ordered) == 2

    def test_combinational_loop_detected(self):
        """a = f(b); b = f(a) → loop."""
        stmts = [
            (0, (Assign(target="a", value=SignalRef(shape=Shape(8), name="b")),)),
            (1, (Assign(target="b", value=SignalRef(shape=Shape(8), name="a")),)),
        ]
        assignments = build_assignments(stmts)
        with pytest.raises(CombLoopError):
            topological_sort(assignments)

    def test_self_reference_not_loop(self):
        """a = a + 1 in a single block is NOT a loop (self-read within same block)."""
        stmts = [
            (
                0,
                (
                    Assign(
                        target="a",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="a"),
                            right=Const(shape=Shape(8), value=1),
                        ),
                    ),
                ),
            ),
        ]
        assignments = build_assignments(stmts)
        # Self-reference within same block is skipped
        ordered = topological_sort(assignments)
        assert len(ordered) == 1

    def test_diamond_dependency(self):
        """a → b, a → c, b+c → d."""
        stmts = [
            (0, (Assign(target="b", value=SignalRef(shape=Shape(8), name="a")),)),
            (1, (Assign(target="c", value=SignalRef(shape=Shape(8), name="a")),)),
            (
                2,
                (
                    Assign(
                        target="d",
                        value=Binary(
                            shape=Shape(8),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(8), name="b"),
                            right=SignalRef(shape=Shape(8), name="c"),
                        ),
                    ),
                ),
            ),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)
        names = [list(a.writes)[0] for a in ordered]
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")


class TestWireResolution:
    def test_single_driver(self):
        v = FourState.from_int(42, Shape(8))
        r = resolve_drivers([v], NetKind.WIRE, Shape(8))
        assert r.to_int == 42

    def test_no_drivers_is_z(self):
        r = resolve_drivers([], NetKind.WIRE, Shape(8))
        assert r.has_unknown
        assert r.aval == 0
        assert r.bval == 0xFF

    def test_tri_z_yields_to_driver(self):
        """Z + defined → defined."""
        d1 = FourState.z(Shape(8))
        d2 = FourState.from_int(0xAB, Shape(8))
        r = resolve_drivers([d1, d2], NetKind.TRI, Shape(8))
        assert r.to_int == 0xAB

    def test_wand(self):
        """Wired-AND: 0xFF & 0x0F = 0x0F."""
        d1 = FourState.from_int(0xFF, Shape(8))
        d2 = FourState.from_int(0x0F, Shape(8))
        r = resolve_drivers([d1, d2], NetKind.WAND, Shape(8))
        assert r.to_int == 0x0F

    def test_wor(self):
        """Wired-OR: 0xF0 | 0x0F = 0xFF."""
        d1 = FourState.from_int(0xF0, Shape(8))
        d2 = FourState.from_int(0x0F, Shape(8))
        r = resolve_drivers([d1, d2], NetKind.WOR, Shape(8))
        assert r.to_int == 0xFF


class TestCombLoopDetection:
    def test_loop_in_module(self):
        """Module with two comb blocks forming a loop should fail to compile."""
        m = Module(
            name="loopy",
            ports=(
                Port(Signal("a", Shape(8)), PortDirection.INPUT),
                Port(Signal("b", Shape(8)), PortDirection.OUTPUT),
                Port(Signal("c", Shape(8)), PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", clk="a"),),
            comb_blocks=(
                CombBlock(stmts=(Assign(target="b", value=SignalRef(shape=Shape(8), name="c")),)),
                CombBlock(stmts=(Assign(target="c", value=SignalRef(shape=Shape(8), name="b")),)),
            ),
        )
        with pytest.raises(CombLoopError):
            compile_module(m)
