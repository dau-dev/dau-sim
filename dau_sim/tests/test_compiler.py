import pytest

from dau_sim.compiler import CombLoopError, compile_module
from dau_sim.compiler.depanalysis import (
    affected_component_ids,
    build_assignments,
    build_component_signal_index,
    collect_reads,
    collect_stmt_writes,
    partition_assignments,
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
        names = [next(iter(a.writes)) for a in ordered]
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
        names = [next(iter(a.writes)) for a in ordered]
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_partition_assignments_parallel_components(self):
        """Independent combinational blocks should land in separate components."""
        stmts = [
            (0, (Assign(target="x", value=SignalRef(shape=Shape(8), name="a")),)),
            (1, (Assign(target="y", value=SignalRef(shape=Shape(8), name="b")),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)

        components = partition_assignments(ordered)

        assert len(components) == 2
        assert {frozenset(component.writes) for component in components} == {frozenset({"x"}), frozenset({"y"})}

    def test_partition_assignments_connected_chain(self):
        """A dependency chain should remain a single component."""
        stmts = [
            (0, (Assign(target="c", value=SignalRef(shape=Shape(8), name="b")),)),
            (1, (Assign(target="b", value=SignalRef(shape=Shape(8), name="a")),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)

        components = partition_assignments(ordered)

        assert len(components) == 1
        assert components[0].writes == frozenset({"b", "c"})

    def test_partition_assignments_shared_write_signal(self):
        """Assignments that write the same signal must stay in one component."""
        stmts = [
            (0, (Assign(target="y", value=SignalRef(shape=Shape(8), name="a")),)),
            (1, (Assign(target="y", value=SignalRef(shape=Shape(8), name="b")),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)

        components = partition_assignments(ordered)

        assert len(components) == 1
        assert components[0].writes == frozenset({"y"})

    def test_build_component_signal_index_maps_signal_to_component_ids(self):
        """Each signal should map to the component ids that reference it."""
        stmts = [
            (0, (Assign(target="x", value=SignalRef(shape=Shape(8), name="a")),)),
            (1, (Assign(target="y", value=SignalRef(shape=Shape(8), name="b")),)),
            (2, (Assign(target="z", value=SignalRef(shape=Shape(8), name="x")),)),
        ]
        assignments = build_assignments(stmts)
        ordered = topological_sort(assignments)
        components = partition_assignments(ordered)

        index = build_component_signal_index(components)

        assert index["a"] == 0b01
        assert index["x"] == 0b01
        assert index["b"] == 0b10
        assert index["y"] == 0b10
        assert index["z"] == 0b01

    def test_affected_component_ids_deduplicates_and_sorts(self):
        """Changed signals should resolve to unique component ids in component order."""
        signal_index = {
            "a": (1 << 2) | (1 << 0),
            "b": (1 << 1),
            "c": (1 << 2),
        }

        affected = affected_component_ids(signal_index, {"a", "b", "c"})

        assert affected == (0, 1, 2)


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
                Port(signal=Signal(name="a", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="b", shape=Shape(8)), direction=PortDirection.OUTPUT),
                Port(signal=Signal(name="c", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain(name="sync", clk="a"),),
            comb_blocks=(
                CombBlock(stmts=(Assign(target="b", value=SignalRef(shape=Shape(8), name="c")),)),
                CombBlock(stmts=(Assign(target="c", value=SignalRef(shape=Shape(8), name="b")),)),
            ),
        )
        with pytest.raises(CombLoopError):
            compile_module(m)


class TestRewriter:
    def test_rewrite_signalref(self):
        expr = SignalRef(shape=Shape(8), name="count")
        result = prefix_expr(expr, "child")
        assert result.name == "child.count"

    def test_rewrite_const_unchanged(self):
        expr = Const(shape=Shape(8), value=42)
        result = prefix_expr(expr, "child")
        assert result is expr

    def test_rewrite_binary_expr(self):
        expr = Binary(shape=Shape(8), op=BinaryOp.ADD, left=SignalRef(shape=Shape(8), name="a"), right=SignalRef(shape=Shape(8), name="b"))
        result = prefix_expr(expr, "sub")
        assert result.left.name == "sub.a"
        assert result.right.name == "sub.b"

    def test_rewrite_unary_expr(self):
        expr = Unary(shape=Shape(8), op=UnaryOp.NOT, operand=SignalRef(shape=Shape(8), name="x"))
        result = prefix_expr(expr, "p")
        assert result.operand.name == "p.x"

    def test_rewrite_mux_expr(self):
        expr = Mux(
            shape=Shape(8),
            sel=SignalRef(shape=Shape(1), name="sel"),
            if_true=SignalRef(shape=Shape(8), name="a"),
            if_false=SignalRef(shape=Shape(8), name="b"),
        )
        result = prefix_expr(expr, "m")
        assert result.sel.name == "m.sel"
        assert result.if_true.name == "m.a"
        assert result.if_false.name == "m.b"

    def test_rewrite_concat_expr(self):
        expr = Concat(shape=Shape(16), parts=(SignalRef(shape=Shape(8), name="hi"), SignalRef(shape=Shape(8), name="lo")))
        result = prefix_expr(expr, "c")
        assert result.parts[0].name == "c.hi"
        assert result.parts[1].name == "c.lo"

    def test_rewrite_slice_expr(self):
        expr = Slice(shape=Shape(4), value=SignalRef(shape=Shape(8), name="val"), low=0, high=4)
        result = prefix_expr(expr, "s")
        assert result.value.name == "s.val"

    def test_rewrite_assign_stmt(self):
        stmt = Assign(target="out", value=SignalRef(shape=Shape(1), name="in_sig"))
        result = prefix_stmt(stmt, "child")
        assert result.target == "child.out"
        assert result.value.name == "child.in_sig"

    def test_rewrite_ifelse_stmt(self):
        stmt = IfElse(
            cond=SignalRef(shape=Shape(1), name="en"),
            then_body=(Assign(target="o", value=SignalRef(shape=Shape(1), name="a")),),
            else_body=(Assign(target="o", value=Const(shape=Shape(1), value=0)),),
        )
        result = prefix_stmt(stmt, "x")
        assert result.cond.name == "x.en"
        assert result.then_body[0].target == "x.o"

    def test_rewrite_switch_stmt(self):
        stmt = IrSwitch(
            test=SignalRef(shape=Shape(2), name="sel"),
            cases=((0, (Assign(target="y", value=Const(shape=Shape(1), value=0)),)),),
        )
        result = prefix_stmt(stmt, "p")
        assert result.test.name == "p.sel"
        assert result.cases[0][1][0].target == "p.y"

    def test_prefix_stmts_batch(self):
        stmts = (
            Assign(target="a", value=Const(shape=Shape(1), value=0)),
            Assign(target="b", value=SignalRef(shape=Shape(1), name="c")),
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
                Port(signal=Signal(name="in1", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="out1", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            signals=(Signal(name="internal", shape=Shape(8)),),
            comb_blocks=(CombBlock(stmts=(Assign(target="out1", value=SignalRef(shape=Shape(8), name="in1")),)),),
        )
        parent = Module(
            name="top",
            ports=(
                Port(signal=Signal(name="top_in", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="top_out", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            instances=(
                Instance(
                    name="u0",
                    module_name="child",
                    bindings=(
                        PortBinding(port_name="in1", expr=SignalRef(shape=Shape(8), name="top_in")),
                        PortBinding(port_name="out1", expr=SignalRef(shape=Shape(8), name="top_out")),
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
            ports=(Port(signal=Signal(name="x", shape=Shape(1)), direction=PortDirection.INPUT),),
        )
        flat = flatten_module(m)
        assert flat.name == "simple"
        assert "x" in flat.all_signal_names

    def test_flatten_preserves_memories(self):
        """Memories from child modules are flattened with prefix."""
        from dau_sim.compiler.flatten import flatten_module

        child = Module(
            name="mem_child",
            memories=(Memory(name="ram", shape=Shape(8), depth=16, read_ports=(), write_ports=()),),
        )
        parent = Module(
            name="top",
            instances=(Instance(name="u_mem", module_name="mem_child", bindings=()),),
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
            signals=(Signal(name="s", shape=Shape(4)),),
        )
        child = Module(
            name="child",
            instances=(Instance(name="gc0", module_name="gc", bindings=()),),
            submodules=(grandchild,),
        )
        parent = Module(
            name="top",
            instances=(Instance(name="c0", module_name="child", bindings=()),),
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
                Port(signal=Signal(name="a", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="b", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            comb_blocks=(CombBlock(stmts=(Assign(target="b", value=SignalRef(shape=Shape(8), name="a")),)),),
        )
        parent = Module(
            name="top",
            ports=(
                Port(signal=Signal(name="x", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="y", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            instances=(
                Instance(
                    name="u0",
                    module_name="passthru",
                    bindings=(
                        PortBinding(port_name="a", expr=SignalRef(shape=Shape(8), name="x")),
                        PortBinding(port_name="b", expr=SignalRef(shape=Shape(8), name="y")),
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
                Port(signal=Signal(name="wr_addr", shape=Shape(2)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="wr_data", shape=Shape(8)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="wr_en", shape=Shape(1)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="rd_addr", shape=Shape(2)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="rd_data", shape=Shape(8)), direction=PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain(name="sync", clk="clk"),),
            signals=(Signal(name="clk", shape=Shape(1)),),
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
                Port(signal=Signal(name="wr_addr", shape=Shape(1)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="wr_data", shape=Shape(16)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="wr_en", shape=Shape(2)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="rd_addr", shape=Shape(1)), direction=PortDirection.INPUT),
                Port(signal=Signal(name="rd_data", shape=Shape(16)), direction=PortDirection.OUTPUT),
            ),
            clock_domains=(ClockDomain(name="sync", clk="clk"),),
            signals=(Signal(name="clk", shape=Shape(1)),),
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
