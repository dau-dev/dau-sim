import pytest

from dau_sim.compiler import CombLoopError, compile_module
from dau_sim.compiler.depanalysis import (
    build_assignments,
    collect_reads,
    collect_stmt_writes,
    topological_sort,
)
from dau_sim.compiler.resolve import resolve_drivers
from dau_sim.compiler.rewrite import prefix_expr, prefix_stmt, prefix_stmts
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
from dau_sim.ir.expr import Concat, Mux, Slice, Unary, UnaryOp
from dau_sim.ir.module import Instance, Memory, PortBinding, ReadPort, WritePort
from dau_sim.ir.stmt import IfElse, Switch as IrSwitch


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


class TestRewriter:
    def test_rewrite_signalref(self):
        expr = SignalRef(Shape(8), "count")
        result = prefix_expr(expr, "child")
        assert result.name == "child.count"

    def test_rewrite_const_unchanged(self):
        expr = Const(Shape(8), 42)
        result = prefix_expr(expr, "child")
        assert result is expr

    def test_rewrite_binary_expr(self):
        expr = Binary(Shape(8), BinaryOp.ADD, SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))
        result = prefix_expr(expr, "sub")
        assert result.left.name == "sub.a"
        assert result.right.name == "sub.b"

    def test_rewrite_unary_expr(self):
        expr = Unary(Shape(8), UnaryOp.NOT, SignalRef(Shape(8), "x"))
        result = prefix_expr(expr, "p")
        assert result.operand.name == "p.x"

    def test_rewrite_mux_expr(self):
        expr = Mux(Shape(8), SignalRef(Shape(1), "sel"), SignalRef(Shape(8), "a"), SignalRef(Shape(8), "b"))
        result = prefix_expr(expr, "m")
        assert result.sel.name == "m.sel"
        assert result.if_true.name == "m.a"
        assert result.if_false.name == "m.b"

    def test_rewrite_concat_expr(self):
        expr = Concat(Shape(16), (SignalRef(Shape(8), "hi"), SignalRef(Shape(8), "lo")))
        result = prefix_expr(expr, "c")
        assert result.parts[0].name == "c.hi"
        assert result.parts[1].name == "c.lo"

    def test_rewrite_slice_expr(self):
        expr = Slice(Shape(4), SignalRef(Shape(8), "val"), 0, 4)
        result = prefix_expr(expr, "s")
        assert result.value.name == "s.val"

    def test_rewrite_assign_stmt(self):
        stmt = Assign(target="out", value=SignalRef(Shape(1), "in_sig"))
        result = prefix_stmt(stmt, "child")
        assert result.target == "child.out"
        assert result.value.name == "child.in_sig"

    def test_rewrite_ifelse_stmt(self):
        stmt = IfElse(
            cond=SignalRef(Shape(1), "en"),
            then_body=(Assign(target="o", value=SignalRef(Shape(1), "a")),),
            else_body=(Assign(target="o", value=Const(Shape(1), 0)),),
        )
        result = prefix_stmt(stmt, "x")
        assert result.cond.name == "x.en"
        assert result.then_body[0].target == "x.o"

    def test_rewrite_switch_stmt(self):
        stmt = IrSwitch(
            test=SignalRef(Shape(2), "sel"),
            cases=((0, (Assign(target="y", value=Const(Shape(1), 0)),)),),
        )
        result = prefix_stmt(stmt, "p")
        assert result.test.name == "p.sel"
        assert result.cases[0][1][0].target == "p.y"

    def test_prefix_stmts_batch(self):
        stmts = (
            Assign(target="a", value=Const(Shape(1), 0)),
            Assign(target="b", value=SignalRef(Shape(1), "c")),
        )
        result = prefix_stmts(stmts, "mod")
        assert result[0].target == "mod.a"
        assert result[1].target == "mod.b"
        assert result[1].value.name == "mod.c"


class TestFlatten:
    def test_flatten_simple_hierarchy(self):
        """Parent with one child → flat module."""
        from dau_sim.compiler.flatten import flatten_module

        child = Module(
            name="child",
            ports=(
                Port(Signal("in1", Shape(8)), PortDirection.INPUT),
                Port(Signal("out1", Shape(8)), PortDirection.OUTPUT),
            ),
            signals=(Signal("internal", Shape(8)),),
            comb_blocks=(CombBlock(stmts=(Assign(target="out1", value=SignalRef(Shape(8), "in1")),)),),
        )
        parent = Module(
            name="top",
            ports=(
                Port(Signal("top_in", Shape(8)), PortDirection.INPUT),
                Port(Signal("top_out", Shape(8)), PortDirection.OUTPUT),
            ),
            instances=(
                Instance(
                    "u0",
                    "child",
                    (
                        PortBinding("in1", SignalRef(Shape(8), "top_in")),
                        PortBinding("out1", SignalRef(Shape(8), "top_out")),
                    ),
                ),
            ),
            submodules=(child,),
        )
        flat = flatten_module(parent)
        assert flat.instances == ()
        assert flat.submodules == ()
        names = flat.all_signal_names
        assert "u0.in1" in names
        assert "u0.out1" in names
        assert "u0.internal" in names

    def test_flatten_no_hierarchy(self):
        """Module with no instances returns unchanged."""
        from dau_sim.compiler.flatten import flatten_module

        m = Module(
            name="simple",
            ports=(Port(Signal("x", Shape(1)), PortDirection.INPUT),),
        )
        flat = flatten_module(m)
        assert flat.name == "simple"
        assert "x" in flat.all_signal_names

    def test_flatten_preserves_memories(self):
        """Memories from child modules are flattened with prefix."""
        from dau_sim.compiler.flatten import flatten_module

        child = Module(
            name="mem_child",
            memories=(Memory("ram", Shape(8), 16, (), ()),),
        )
        parent = Module(
            name="top",
            instances=(Instance("u_mem", "mem_child", ()),),
            submodules=(child,),
        )
        flat = flatten_module(parent)
        assert len(flat.memories) == 1
        assert flat.memories[0].name == "u_mem.ram"

    def test_flatten_nested_hierarchy(self):
        """Grandchild is correctly flattened through two levels."""
        from dau_sim.compiler.flatten import flatten_module

        grandchild = Module(
            name="gc",
            signals=(Signal("s", Shape(4)),),
        )
        child = Module(
            name="child",
            instances=(Instance("gc0", "gc", ()),),
            submodules=(grandchild,),
        )
        parent = Module(
            name="top",
            instances=(Instance("c0", "child", ()),),
            submodules=(child,),
        )
        flat = flatten_module(parent)
        names = flat.all_signal_names
        assert "c0.gc0.s" in names

    def test_flatten_port_bindings_create_assignments(self):
        """Port bindings produce combinational assignments."""
        from dau_sim.compiler.flatten import flatten_module

        child = Module(
            name="passthru",
            ports=(
                Port(Signal("a", Shape(8)), PortDirection.INPUT),
                Port(Signal("b", Shape(8)), PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign(target="b", value=SignalRef(Shape(8), "a")),)),),
        )
        parent = Module(
            name="top",
            ports=(
                Port(Signal("x", Shape(8)), PortDirection.INPUT),
                Port(Signal("y", Shape(8)), PortDirection.OUTPUT),
            ),
            instances=(
                Instance(
                    "u0",
                    "passthru",
                    (
                        PortBinding("a", SignalRef(Shape(8), "x")),
                        PortBinding("b", SignalRef(Shape(8), "y")),
                    ),
                ),
            ),
            submodules=(child,),
        )
        flat = flatten_module(parent)
        # Should have comb assignments for port bindings
        all_stmts = []
        for cb in flat.comb_blocks:
            all_stmts.extend(cb.stmts)
        targets = {s.target for s in all_stmts if isinstance(s, Assign)}
        # Input binding: x → u0.a
        assert "u0.a" in targets
        # Output binding: u0.b → y
        assert "y" in targets


class TestMemoryExecution:
    """Tests for memory read/write execution in compiled simulation."""

    def _make_mem_module(self, *, init=(), depth=4, comb_read=True):
        """Build a module with one 8-bit memory, 1 write port, 1 read port.

        Signals: wr_addr, wr_data, wr_en (write), rd_addr, rd_data (read).
        Write domain = 'sync', read domain = None (comb) or 'sync'.
        """
        from dau_sim.ir.module import Memory

        mem = Memory(
            name="mem0",
            shape=Shape(8),
            depth=depth,
            read_ports=(
                ReadPort(
                    addr="rd_addr",
                    data="rd_data",
                    domain=None if comb_read else "sync",
                ),
            ),
            write_ports=(WritePort(addr="wr_addr", data="wr_data", en="wr_en", domain="sync"),),
            init=init,
        )
        return Module(
            name="mem_test",
            ports=(
                Port(Signal("wr_addr", Shape(2)), PortDirection.INPUT),
                Port(Signal("wr_data", Shape(8)), PortDirection.INPUT),
                Port(Signal("wr_en", Shape(1)), PortDirection.INPUT),
                Port(Signal("rd_addr", Shape(2)), PortDirection.INPUT),
                Port(Signal("rd_data", Shape(8)), PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", clk="clk"),),
            signals=(Signal("clk", Shape(1)),),
            memories=(mem,),
        )

    def test_memory_init_values(self):
        """Memory with init data should be readable after compile."""
        mod = self._make_mem_module(init=(0xAA, 0xBB, 0xCC, 0xDD))
        cm = compile_module(mod)
        assert cm._mem_init["mem0"] == [0xAA, 0xBB, 0xCC, 0xDD]

    def test_memory_init_padded(self):
        """Partial init should be zero-padded to depth."""
        mod = self._make_mem_module(init=(0x11,), depth=4)
        cm = compile_module(mod)
        assert cm._mem_init["mem0"] == [0x11, 0, 0, 0]

    def test_memory_shapes_registered(self):
        """Memory port signals should be registered in shapes."""
        mod = self._make_mem_module()
        cm = compile_module(mod)
        assert "rd_addr" in cm._shapes
        assert "rd_data" in cm._shapes
        assert "wr_addr" in cm._shapes
        assert "wr_data" in cm._shapes
        assert "wr_en" in cm._shapes
        # rd_data should have mem element shape
        assert cm._shapes["rd_data"] == Shape(8)

    def test_memory_write_then_comb_read(self):
        """Write to memory, then read back combinationally."""
        mod = self._make_mem_module(init=(0, 0, 0, 0))
        cm = compile_module(mod)
        # Drive wr_en=1, wr_addr=1, wr_data=0x42, rd_addr=1
        traces = cm.run(
            cycles=5,
            inputs={"wr_en": 1, "wr_addr": 1, "wr_data": 0x42, "rd_addr": 1},
        )
        # After a clock edge fires the write and comb read settles,
        # rd_data should eventually show 0x42
        rd_vals = [v for _, v in traces["rd_data"]]
        assert 0x42 in rd_vals

    def test_memory_read_init_data(self):
        """Combinational read should return init data from cycle 1."""
        mod = self._make_mem_module(init=(0xDE, 0xAD, 0xBE, 0xEF))
        cm = compile_module(mod)
        traces = cm.run(cycles=3, inputs={"rd_addr": 2})
        rd_vals = [v for _, v in traces["rd_data"]]
        # Address 2 init = 0xBE
        assert 0xBE in rd_vals

    def test_memory_sync_read(self):
        """Synchronous read port should latch data on clock edge."""
        mod = self._make_mem_module(init=(0x10, 0x20, 0x30, 0x40), comb_read=False)
        cm = compile_module(mod)
        traces = cm.run(cycles=3, inputs={"rd_addr": 3})
        rd_vals = [v for _, v in traces["rd_data"]]
        # Address 3 init = 0x40, should appear after clock edge
        assert 0x40 in rd_vals

    def test_memory_granular_write(self):
        """Granular write enable should only update selected bytes."""
        from dau_sim.ir.module import Memory

        mem = Memory(
            name="mem0",
            shape=Shape(16),
            depth=2,
            read_ports=(ReadPort(addr="rd_addr", data="rd_data", domain=None),),
            write_ports=(
                WritePort(
                    addr="wr_addr",
                    data="wr_data",
                    en="wr_en",
                    domain="sync",
                    granularity=8,
                ),
            ),
            init=(0xAABB, 0),
        )
        mod = Module(
            name="gran_test",
            ports=(
                Port(Signal("wr_addr", Shape(1)), PortDirection.INPUT),
                Port(Signal("wr_data", Shape(16)), PortDirection.INPUT),
                Port(Signal("wr_en", Shape(2)), PortDirection.INPUT),
                Port(Signal("rd_addr", Shape(1)), PortDirection.INPUT),
                Port(Signal("rd_data", Shape(16)), PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain("sync", clk="clk"),),
            signals=(Signal("clk", Shape(1)),),
            memories=(mem,),
        )
        cm = compile_module(mod)
        # Write only high byte (en=0b10) with data=0xFF00 to addr 0
        # Init[0] = 0xAABB → should become 0xFFBB
        traces = cm.run(
            cycles=5,
            inputs={"wr_en": 0b10, "wr_addr": 0, "wr_data": 0xFF00, "rd_addr": 0},
        )
        rd_vals = [v for _, v in traces["rd_data"]]
        assert 0xFFBB in rd_vals
