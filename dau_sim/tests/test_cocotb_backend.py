"""Tests for the cocotb backend (pure-Python GPI replacement).

Tests cover:
- DauSimHandle (signal read/write, child discovery, type introspection)
- DauSimCallback (allocation, deregistration)
- SimulationEngine (init, callback queue, phase ordering, settling, VCH)
- Full integration with cocotb scheduler
"""

from dau_sim.backends.cocotb_backend import (
    _PHASE_NORMAL,
    _PHASE_READONLY,
    _PHASE_READWRITE,
    FALLING,
    LOGIC,
    LOGIC_ARRAY,
    MODULE,
    OBJECTS,
    RANGE_DOWN,
    RANGE_NO_DIR,
    RISING,
    UNKNOWN,
    VALUE_CHANGE,
    DauSimCallback,
    DauSimHandle,
    DauSimIterator,
    SimulationEngine,
    _create_simulator_module,
)
from dau_sim.ir import (
    Assign,
    Binary,
    BinaryOp,
    ClockDomain,
    CombBlock,
    Const,
    EdgePolarity,
    IfElse,
    Module,
    Port,
    PortDirection,
    SeqBlock,
    Shape,
    Signal,
    SignalRef,
)

# Test-fixture helpers


def _make_counter_module(width=4, name="counter"):
    """4-bit counter with rst (sync domain, posedge)."""
    return Module(
        name=name,
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("rst", Shape(1)), PortDirection.INPUT),
            Port(Signal("count", Shape(width)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    IfElse(
                        cond=SignalRef(shape=Shape(1), name="rst"),
                        then_body=(Assign(target="count", value=Const(shape=Shape(width), value=0)),),
                        else_body=(
                            Assign(
                                target="count",
                                value=Binary(
                                    shape=Shape(width),
                                    op=BinaryOp.ADD,
                                    left=SignalRef(shape=Shape(width), name="count"),
                                    right=Const(shape=Shape(width), value=1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_comb_module():
    """Simple combinational: out = a & b."""
    return Module(
        name="comb_and",
        ports=(
            Port(Signal("a", Shape(1)), PortDirection.INPUT),
            Port(Signal("b", Shape(1)), PortDirection.INPUT),
            Port(Signal("out", Shape(1)), PortDirection.OUTPUT),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="out",
                        value=Binary(
                            shape=Shape(1),
                            op=BinaryOp.AND,
                            left=SignalRef(shape=Shape(1), name="a"),
                            right=SignalRef(shape=Shape(1), name="b"),
                        ),
                    ),
                ),
            ),
        ),
    )


class TestDauSimHandle:
    def test_scope_handle_type(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "top", is_scope=True)
        assert h.get_type() == MODULE
        assert h.get_type_string() == "GPI_MODULE"

    def test_logic_handle_type(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "a", shape=Shape(1))
        assert h.get_type() == LOGIC
        assert h.get_type_string() == "GPI_LOGIC"

    def test_logic_array_handle_type(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "count", shape=Shape(4))
        assert h.get_type() == LOGIC_ARRAY
        assert h.get_type_string() == "GPI_LOGIC_ARRAY"

    def test_unknown_handle_type(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "mystery")
        assert h.get_type() == UNKNOWN

    def test_get_name_string(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "sig_a", "top.sig_a")
        assert h.get_name_string() == "sig_a"

    def test_definition(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "sig")
        assert h.get_definition_name() == "sig"
        assert h.get_definition_file() == ""

    def test_const(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "x", is_const=True)
        assert h.get_const() is True
        h2 = DauSimHandle(engine, "y")
        assert h2.get_const() is False

    def test_indexable(self):
        engine = SimulationEngine(_make_comb_module())
        h1 = DauSimHandle(engine, "x", shape=Shape(1))
        assert h1.get_indexable() is False
        h4 = DauSimHandle(engine, "y", shape=Shape(4))
        assert h4.get_indexable() is True

    def test_num_elems_signal(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "x", shape=Shape(8))
        assert h.get_num_elems() == 8

    def test_num_elems_scope(self):
        engine = SimulationEngine(_make_comb_module())
        child_a = DauSimHandle(engine, "a", shape=Shape(1))
        child_b = DauSimHandle(engine, "b", shape=Shape(1))
        h = DauSimHandle(engine, "top", is_scope=True, children={"a": child_a, "b": child_b})
        assert h.get_num_elems() == 2

    def test_range_multibit(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "x", shape=Shape(8))
        assert h.get_range() == (7, 0, RANGE_DOWN)

    def test_range_singlebit(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "x", shape=Shape(1))
        assert h.get_range() == (0, 0, RANGE_NO_DIR)

    def test_child_discovery(self):
        engine = SimulationEngine(_make_comb_module())
        child = DauSimHandle(engine, "child", shape=Shape(1))
        parent = DauSimHandle(engine, "top", is_scope=True, children={"child": child})
        assert parent.get_handle_by_name("child") is child
        assert parent.get_handle_by_name("missing") is None

    def test_bit_index(self):
        engine = SimulationEngine(_make_comb_module())
        engine._signals["wide"] = 0b1010
        engine._shapes["wide"] = Shape(4)
        h = DauSimHandle(engine, "wide", shape=Shape(4))
        bit1 = h.get_handle_by_index(1)
        assert bit1.get_signal_val_long() == 1  # bit 1 of 0b1010
        bit0 = h.get_handle_by_index(0)
        assert bit0.get_signal_val_long() == 0  # bit 0 of 0b1010

    def test_iterate(self):
        engine = SimulationEngine(_make_comb_module())
        c1 = DauSimHandle(engine, "a", shape=Shape(1))
        c2 = DauSimHandle(engine, "b", shape=Shape(1))
        parent = DauSimHandle(engine, "top", is_scope=True, children={"a": c1, "b": c2})
        items = list(parent.iterate(OBJECTS))
        assert len(items) == 2

    def test_signal_read_write(self):
        engine = SimulationEngine(_make_comb_module())
        engine._signals["x"] = 0
        engine._shapes["x"] = Shape(8)
        h = DauSimHandle(engine, "x", shape=Shape(8))
        assert h.get_signal_val_long() == 0
        h.set_signal_val_int(0, 42)
        assert h.get_signal_val_long() == 42

    def test_signal_binstr(self):
        engine = SimulationEngine(_make_comb_module())
        engine._signals["x"] = 5
        engine._shapes["x"] = Shape(4)
        h = DauSimHandle(engine, "x", shape=Shape(4))
        assert h.get_signal_val_binstr() == "0101"
        h.set_signal_val_binstr(0, "1100")
        assert h.get_signal_val_long() == 12

    def test_signal_real(self):
        engine = SimulationEngine(_make_comb_module())
        engine._signals["x"] = 7
        engine._shapes["x"] = Shape(8)
        h = DauSimHandle(engine, "x", shape=Shape(8))
        assert h.get_signal_val_real() == 7.0

    def test_bit_write(self):
        engine = SimulationEngine(_make_comb_module())
        engine._signals["v"] = 0b0000
        engine._shapes["v"] = Shape(4)
        h = DauSimHandle(engine, "v", shape=Shape(4))
        bit2 = h.get_handle_by_index(2)
        bit2.set_signal_val_int(0, 1)
        assert engine._signals["v"] == 0b0100

    def test_hash_and_eq(self):
        engine = SimulationEngine(_make_comb_module())
        h1 = DauSimHandle(engine, "a")
        h2 = DauSimHandle(engine, "a")
        assert h1 != h2  # identity-based
        assert h1 == h1
        assert hash(h1) != hash(h2)


class TestDauSimIterator:
    def test_basic_iteration(self):
        engine = SimulationEngine(_make_comb_module())
        handles = [DauSimHandle(engine, f"s{i}") for i in range(3)]
        it = DauSimIterator(handles)
        result = list(it)
        assert result == handles

    def test_empty(self):
        it = DauSimIterator([])
        assert list(it) == []


class TestDauSimCallback:
    def test_alloc_and_deregister(self):
        engine = SimulationEngine(_make_comb_module())
        cb_id = engine._alloc_cb_id()
        assert cb_id in engine._active_callbacks
        cb = DauSimCallback(engine, cb_id)
        cb.deregister()
        assert cb_id not in engine._active_callbacks

    def test_double_deregister(self):
        engine = SimulationEngine(_make_comb_module())
        cb_id = engine._alloc_cb_id()
        cb = DauSimCallback(engine, cb_id)
        cb.deregister()
        cb.deregister()  # should not raise

    def test_deregister_removes_vch(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "s", shape=Shape(1))
        cb = engine.register_value_change_callback(h, lambda *a: None, RISING)
        assert "s" in engine._vch_callbacks
        cb.deregister()
        assert "s" not in engine._vch_callbacks


class TestEngineInit:
    def test_counter_module_signals(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        assert "clk" in engine._signals
        assert "rst" in engine._signals
        assert "count" in engine._signals
        assert engine._signals["count"] == 0

    def test_comb_module_signals(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        assert "a" in engine._signals
        assert "b" in engine._signals
        assert "out" in engine._signals

    def test_domain_info_built(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        assert "sync" in engine._domain_info
        di = engine._domain_info["sync"]
        assert di["clk_signal"] == "clk"
        assert di["edge"] == EdgePolarity.POSEDGE

    def test_has_seq_flag(self):
        assert SimulationEngine(_make_counter_module())._has_seq is True
        assert SimulationEngine(_make_comb_module())._has_seq is False

    def test_comb_initial_settle(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        # a=0, b=0 -> out=0
        assert engine._signals["out"] == 0


class TestEngineCallbacks:
    def test_timed_callback(self):
        engine = SimulationEngine(_make_comb_module())
        fired = []
        engine.register_timed_callback(100, lambda *a: fired.append(a), "arg")
        assert len(engine._callback_queue) == 1
        t, phase, _, _, _, _ = engine._callback_queue[0]
        assert t == 100
        assert phase == _PHASE_NORMAL

    def test_readonly_callback(self):
        engine = SimulationEngine(_make_comb_module())
        engine.register_readonly_callback(lambda *a: None)
        _, phase, _, _, _, _ = engine._callback_queue[0]
        assert phase == _PHASE_READONLY

    def test_rwsynch_callback(self):
        engine = SimulationEngine(_make_comb_module())
        engine.register_rwsynch_callback(lambda *a: None)
        _, phase, _, _, _, _ = engine._callback_queue[0]
        assert phase == _PHASE_READWRITE

    def test_nextstep_callback(self):
        engine = SimulationEngine(_make_comb_module())
        engine.register_nextstep_callback(lambda *a: None)
        t, phase, _, _, _, _ = engine._callback_queue[0]
        assert t == 1  # sim_time starts at 0, next step is 1
        assert phase == _PHASE_NORMAL

    def test_value_change_callback(self):
        engine = SimulationEngine(_make_comb_module())
        h = DauSimHandle(engine, "a", shape=Shape(1))
        engine.register_value_change_callback(h, lambda *a: None, RISING)
        assert "a" in engine._vch_callbacks
        assert len(engine._vch_callbacks["a"]) == 1


class TestEngineStepping:
    def test_step_advances_time(self):
        engine = SimulationEngine(_make_comb_module())
        engine.register_timed_callback(50, lambda *a: None)
        engine.step()
        assert engine._sim_time == 50

    def test_step_fires_timed_callback(self):
        engine = SimulationEngine(_make_comb_module())
        fired = []
        engine.register_timed_callback(10, lambda *a: fired.append("ok"))
        engine.step()
        assert fired == ["ok"]

    def test_step_returns_false_when_empty(self):
        engine = SimulationEngine(_make_comb_module())
        assert engine.step() is False

    def test_callbacks_fire_in_phase_order(self):
        """At the same time step, Normal fires before ReadWrite before ReadOnly."""
        engine = SimulationEngine(_make_comb_module())
        order = []
        # Register in reverse order to verify sorting
        engine.register_readonly_callback(lambda *a: order.append("ro"))
        engine.register_rwsynch_callback(lambda *a: order.append("rw"))
        engine.register_timed_callback(0, lambda *a: order.append("normal"))
        # All at time 0 — but step() won't pick time 0 if sim_time == 0
        # because the queue has time==0 entries, step advances to earliest = 0
        engine.step()
        assert order == ["normal", "rw", "ro"]

    def test_deregistered_callback_not_fired(self):
        engine = SimulationEngine(_make_comb_module())
        fired = []
        cb = engine.register_timed_callback(10, lambda *a: fired.append("x"))
        cb.deregister()
        engine.step()
        assert fired == []

    def test_settle_detects_posedge(self):
        """When clk goes 0→1 via set_signal, _settle detects the edge and runs seq blocks."""
        m = _make_counter_module()
        engine = SimulationEngine(m)
        assert engine._signals["count"] == 0
        # Simulate a rising edge: clk was 0, set to 1
        engine.set_signal("clk", 1)
        engine._settle()
        # Counter should have incremented
        assert engine._signals["count"] == 1
        engine._prev_signals = dict(engine._signals)
        # No edge on second settle (clk stays at 1)
        engine._settle()
        assert engine._signals["count"] == 1

    def test_settle_no_edge_on_negedge_for_posedge_domain(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        # First rising edge
        engine.set_signal("clk", 1)
        engine._settle()
        assert engine._signals["count"] == 1
        engine._prev_signals = dict(engine._signals)
        # Falling edge - posedge domain should NOT fire
        engine.set_signal("clk", 0)
        engine._settle()
        assert engine._signals["count"] == 1

    def test_comb_settle(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        engine.set_signal("a", 1)
        engine.set_signal("b", 1)
        engine._settle()
        assert engine._signals["out"] == 1
        engine.set_signal("a", 0)
        engine._settle()
        assert engine._signals["out"] == 0


class TestEngineVCH:
    def test_rising_edge_fires(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        h = DauSimHandle(engine, "clk", shape=Shape(1))
        fired = []
        engine.register_value_change_callback(h, lambda *a: fired.append("rise"), RISING)

        # Simulate clk 0→1
        engine.set_signal("clk", 1)
        engine._check_value_changes()
        assert fired == ["rise"]

    def test_falling_edge_fires(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        engine.set_signal("clk", 1)
        engine._prev_signals = dict(engine._signals)
        h = DauSimHandle(engine, "clk", shape=Shape(1))
        fired = []
        engine.register_value_change_callback(h, lambda *a: fired.append("fall"), FALLING)

        # Simulate clk 1→0
        engine.set_signal("clk", 0)
        engine._check_value_changes()
        assert fired == ["fall"]

    def test_value_change_fires(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        h = DauSimHandle(engine, "a", shape=Shape(1))
        fired = []
        engine.register_value_change_callback(h, lambda *a: fired.append("vc"), VALUE_CHANGE)

        engine.set_signal("a", 1)
        engine._check_value_changes()
        assert fired == ["vc"]

    def test_vch_one_shot(self):
        """VCH callbacks fire once and are removed."""
        m = _make_comb_module()
        engine = SimulationEngine(m)
        h = DauSimHandle(engine, "a", shape=Shape(1))
        fired = []
        engine.register_value_change_callback(h, lambda *a: fired.append(1), VALUE_CHANGE)

        engine.set_signal("a", 1)
        engine._check_value_changes()
        engine._prev_signals = dict(engine._signals)
        engine.set_signal("a", 0)
        engine._check_value_changes()
        assert fired == [1]  # only 1 fire, not 2

    def test_vch_no_fire_on_no_change(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        h = DauSimHandle(engine, "a", shape=Shape(1))
        fired = []
        engine.register_value_change_callback(h, lambda *a: fired.append(1), VALUE_CHANGE)
        # a stays at 0
        engine._check_value_changes()
        assert fired == []


class TestEngineFullStep:
    def test_timed_callback_then_readwrite(self):
        """Simulates the cocotb flow: Timer → write scheduler → ReadWrite."""
        m = _make_counter_module()
        engine = SimulationEngine(m)

        order = []

        def timer_cb(*args):
            # Simulate what cocotb does: schedule a signal write and a ReadWrite
            order.append("timer")
            engine.set_signal("clk", 1)
            engine.register_rwsynch_callback(lambda *a: order.append("rw"))

        engine.register_timed_callback(100, timer_cb)
        engine.step()

        assert engine._sim_time == 100
        assert "timer" in order
        assert "rw" in order
        # After step, clk should be 1 and count should have incremented
        assert engine._signals["clk"] == 1
        assert engine._signals["count"] == 1

    def test_multiple_time_steps(self):
        """Simulate multiple clock toggles."""
        m = _make_counter_module()
        engine = SimulationEngine(m)

        # Manually toggle the clock each step
        for cycle in range(5):
            # Rising edge at time cycle*2
            engine.register_timed_callback(1, lambda *a: engine.set_signal("clk", 1))
            engine.step()
            engine._prev_signals = dict(engine._signals)
            # Falling edge
            engine.register_timed_callback(1, lambda *a: engine.set_signal("clk", 0))
            engine.step()
            engine._prev_signals = dict(engine._signals)

        # After 5 cycles, count should be 5
        assert engine._signals["count"] == 5

    def test_run_with_max_steps(self):
        m = _make_comb_module()
        engine = SimulationEngine(m)
        engine.register_timed_callback(1, lambda *a: None)
        engine.run(max_steps=1)
        assert engine._sim_time == 1


class TestHandleHierarchy:
    def test_build_hierarchy(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        root = engine.build_handle_hierarchy()
        assert root.get_type() == MODULE
        assert root.get_name_string() == "counter"

    def test_children_accessible(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        root = engine.build_handle_hierarchy()
        clk = root.get_handle_by_name("clk")
        assert clk is not None
        assert clk.get_type() == LOGIC
        count = root.get_handle_by_name("count")
        assert count is not None
        assert count.get_type() == LOGIC_ARRAY

    def test_iteration_over_children(self):
        m = _make_counter_module()
        engine = SimulationEngine(m)
        root = engine.build_handle_hierarchy()
        children = list(root.iterate(OBJECTS))
        names = {c.get_name_string() for c in children}
        assert "clk" in names
        assert "rst" in names
        assert "count" in names


class TestSimulatorModule:
    def test_constants(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        assert mod.MODULE == MODULE
        assert mod.LOGIC == LOGIC
        assert mod.RISING == RISING

    def test_types(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        assert mod.gpi_sim_hdl is DauSimHandle
        assert mod.gpi_cb_hdl is DauSimCallback

    def test_get_root_handle(self):
        engine = SimulationEngine(_make_comb_module())
        engine.build_handle_hierarchy()
        mod = _create_simulator_module(engine)
        root = mod.get_root_handle("top")
        assert root is engine._root_handle

    def test_get_sim_time(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        lo, hi = mod.get_sim_time()
        assert lo == 0
        assert hi == 0
        engine._sim_time = 42
        lo, hi = mod.get_sim_time()
        assert lo == 42
        assert hi == 0

    def test_get_precision(self):
        engine = SimulationEngine(_make_comb_module(), time_precision=-12)
        mod = _create_simulator_module(engine)
        assert mod.get_precision() == -12

    def test_product_and_version(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        assert mod.get_simulator_product() == "dau-sim"
        assert mod.get_simulator_version() == "0.1.0"

    def test_stop_simulator(self):
        engine = SimulationEngine(_make_comb_module())
        engine._running = True
        mod = _create_simulator_module(engine)
        mod.stop_simulator()
        assert engine._running is False

    def test_register_timed(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        cb = mod.register_timed_callback(50, lambda *a: None)
        assert isinstance(cb, DauSimCallback)
        assert len(engine._callback_queue) == 1

    def test_register_value_change(self):
        engine = SimulationEngine(_make_comb_module())
        engine.build_handle_hierarchy()
        mod = _create_simulator_module(engine)
        h = engine._root_handle.get_handle_by_name("a")
        cb = mod.register_value_change_callback(h, lambda *a: None, RISING)
        assert isinstance(cb, DauSimCallback)

    def test_package_iterate_returns_none(self):
        engine = SimulationEngine(_make_comb_module())
        mod = _create_simulator_module(engine)
        assert mod.package_iterate() is None


class TestCocotbIntegration:
    """Integration tests that exercise the cocotb scheduler with our engine.

    These tests patch cocotb.simulator and run the scheduler directly,
    verifying that the Timer / RisingEdge / ReadWrite flow works.
    """

    def _setup_cocotb(self, module):
        """Patch cocotb.simulator and return (engine, scheduler)."""
        import logging
        import random
        import sys
        import types

        import cocotb

        # isort: off
        import cocotb.handle  # must precede cocotb._gpi_triggers (circular import)
        import cocotb._gpi_triggers
        import cocotb.simtime
        # isort: on

        engine = SimulationEngine(module)
        engine.build_handle_hierarchy()
        sim_module = _create_simulator_module(engine)

        # Save
        self._orig_sim = sys.modules.get("cocotb.simulator")
        self._orig_cocotb_sim = getattr(cocotb, "simulator", None)
        self._orig_triggers_sim = getattr(cocotb._gpi_triggers, "simulator", None)
        self._orig_handle_sim = getattr(cocotb.handle, "simulator", None)
        self._orig_simtime_sim = getattr(cocotb.simtime, "simulator", None)
        self._orig_is_sim = getattr(cocotb, "is_simulation", False)
        self._orig_top = getattr(cocotb, "top", None)
        self._orig_sched = getattr(cocotb, "_scheduler_inst", None)
        self._orig_regression = getattr(cocotb, "_regression_manager", None)

        # Patch
        sys.modules["cocotb.simulator"] = sim_module
        cocotb.simulator = sim_module
        cocotb._gpi_triggers.simulator = sim_module
        cocotb.handle.simulator = sim_module
        cocotb.simtime.simulator = sim_module
        cocotb.is_simulation = True
        cocotb.argv = []
        cocotb.plusargs = {}
        cocotb.SIM_NAME = "dau-sim"
        cocotb.SIM_VERSION = "0.1.0"
        cocotb.RANDOM_SEED = 0
        random.seed(0)
        cocotb.packages = types.SimpleNamespace()
        if not hasattr(cocotb, "log") or cocotb.log is None:
            cocotb.log = logging.getLogger("test")
            cocotb.log.setLevel(logging.INFO)
        cocotb.top = cocotb.handle._make_sim_object(engine._root_handle)
        cocotb.simtime._init()

        from cocotb._scheduler import Scheduler

        cocotb._scheduler_inst = Scheduler()

        # Minimal regression manager stub so start_soon/create_task works
        from cocotb._test import RunningTest
        from cocotb.regression import RegressionManager

        cocotb._regression_manager = RegressionManager.__new__(RegressionManager)
        cocotb._regression_manager._running_test = RunningTest.__new__(RunningTest)
        cocotb._regression_manager._running_test.tasks = []
        cocotb._regression_manager._running_test._main_task = None

        return engine

    def _teardown_cocotb(self):
        import sys  # isort: skip

        import cocotb

        # isort: off
        import cocotb.handle  # must precede cocotb._gpi_triggers (circular import)
        import cocotb._gpi_triggers
        import cocotb.simtime
        # isort: on

        cocotb.handle._handle2obj.clear()
        if self._orig_sim is not None:
            sys.modules["cocotb.simulator"] = self._orig_sim
            cocotb.simulator = self._orig_cocotb_sim
        else:
            sys.modules.pop("cocotb.simulator", None)
        cocotb._gpi_triggers.simulator = self._orig_triggers_sim
        cocotb.handle.simulator = self._orig_handle_sim
        cocotb.simtime.simulator = self._orig_simtime_sim
        cocotb.is_simulation = self._orig_is_sim
        cocotb.top = self._orig_top
        cocotb._scheduler_inst = self._orig_sched
        cocotb._regression_manager = self._orig_regression

    def test_timer_trigger(self):
        """A Timer(100) trigger should fire after 100 sim time steps."""
        import cocotb

        engine = self._setup_cocotb(_make_comb_module())
        try:
            from cocotb._gpi_triggers import Timer
            from cocotb.task import Task

            result = []

            async def test_coro():
                await Timer(100, unit="step")
                result.append(engine._sim_time)

            task = Task(test_coro())
            cocotb._scheduler_inst._schedule_task_internal(task)
            cocotb._scheduler_inst._event_loop()

            # After event_loop, the task is suspended waiting for Timer(100)
            # The engine has a callback at time 100
            assert len(engine._callback_queue) >= 1

            engine._running = True
            engine.run(max_steps=200)

            assert len(result) == 1
            assert result[0] == 100
        finally:
            self._teardown_cocotb()

    def test_rising_edge_trigger(self):
        """RisingEdge should fire when signal goes 0→1."""
        import cocotb

        engine = self._setup_cocotb(_make_counter_module())
        try:
            from cocotb._gpi_triggers import RisingEdge, Timer
            from cocotb.task import Task

            result = []

            async def clock_driver():
                """Toggle clk via Timer."""
                for _ in range(3):
                    cocotb.top.clk.value = 1
                    await Timer(5000, unit="step")
                    cocotb.top.clk.value = 0
                    await Timer(5000, unit="step")

            async def test_coro():
                await RisingEdge(cocotb.top.clk)
                result.append(engine._signals["count"])

            t1 = Task(clock_driver())
            t2 = Task(test_coro())
            cocotb._scheduler_inst._schedule_task_internal(t1)
            cocotb._scheduler_inst._schedule_task_internal(t2)

            # Start write scheduler (needed for signal writes)
            cocotb.handle._start_write_scheduler()

            cocotb._scheduler_inst._event_loop()

            engine._running = True
            engine.run(max_steps=100_000)

            cocotb.handle._stop_write_scheduler()

            # RisingEdge fires *before* NBA — test sees pre-NBA value
            assert len(result) == 1
            assert result[0] == 0
        finally:
            self._teardown_cocotb()

    def test_counter_via_cocotb_clock(self):
        """Use cocotb.clock.Clock to drive the counter and verify count increments."""
        import cocotb

        engine = self._setup_cocotb(_make_counter_module())
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock
            from cocotb.task import Task

            results = []

            async def test_coro():
                clock = Clock(cocotb.top.clk, 10, unit="ns")
                clock.start()

                # Wait for a few clock cycles
                for _ in range(5):
                    await RisingEdge(cocotb.top.clk)
                    results.append(engine._signals["count"])

                engine.stop()

            task = Task(test_coro())
            cocotb._scheduler_inst._schedule_task_internal(task)
            cocotb.handle._start_write_scheduler()
            cocotb._scheduler_inst._event_loop()

            engine._running = True
            engine.run(max_steps=10_000_000)

            cocotb.handle._stop_write_scheduler()

            assert len(results) == 5
            # RisingEdge fires *before* NBA — values lag by one cycle
            assert results == [0, 1, 2, 3, 4]
        finally:
            self._teardown_cocotb()

    def test_combinational_read(self):
        """Verify that combinational logic settles after signal writes."""
        import cocotb

        engine = self._setup_cocotb(_make_comb_module())
        try:
            from cocotb._gpi_triggers import Timer
            from cocotb.task import Task

            results = []

            async def test_coro():
                cocotb.top.a.value = 1
                cocotb.top.b.value = 1
                await Timer(10, unit="step")
                results.append(engine._signals["out"])

                cocotb.top.a.value = 0
                await Timer(10, unit="step")
                results.append(engine._signals["out"])

                engine.stop()

            task = Task(test_coro())
            cocotb._scheduler_inst._schedule_task_internal(task)
            cocotb.handle._start_write_scheduler()
            cocotb._scheduler_inst._event_loop()

            engine._running = True
            engine.run(max_steps=100_000)

            cocotb.handle._stop_write_scheduler()

            assert results == [1, 0]
        finally:
            self._teardown_cocotb()

    def test_timer_and_edge_same_timestep(self):
        """Timer+edge combination should complete when both happen together."""
        import cocotb

        engine = self._setup_cocotb(_make_comb_module())
        try:
            from cocotb._gpi_triggers import RisingEdge, Timer
            from cocotb.task import Task
            from cocotb.triggers import Combine

            result = []

            async def driver():
                # Schedule a rising edge at the exact same time as Timer(100).
                await Timer(100, unit="step")
                cocotb.top.a.value = 1

            async def waiter():
                await Combine(
                    Timer(100, unit="step"),
                    RisingEdge(cocotb.top.a),
                )
                result.append(engine._sim_time)
                engine.stop()

            t1 = Task(driver())
            t2 = Task(waiter())
            cocotb._scheduler_inst._schedule_task_internal(t1)
            cocotb._scheduler_inst._schedule_task_internal(t2)
            cocotb.handle._start_write_scheduler()
            cocotb._scheduler_inst._event_loop()

            engine._running = True
            engine.run(max_steps=100_000)

            cocotb.handle._stop_write_scheduler()

            assert len(result) == 1
            assert result[0] == 100
        finally:
            self._teardown_cocotb()
