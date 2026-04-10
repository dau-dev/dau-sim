"""End-to-end tests based on cocotb's official examples.

Each test translates a cocotb example HDL design into dau-sim IR, sets up
the cocotb scheduler with a patched simulator module, and runs the same async
test patterns used in the upstream cocotb examples — verifying that dau-sim
drives the cocotb testbench correctly without any intermediary compilation step.

Covered examples:
- simple_dff: D flip-flop with clock-driven sequential test
- adder: Purely combinational adder with reference-model checks
- simple_counter: Counter with rst/ena/set/din, multi-coroutine tests
"""

import logging
import random
import sys
import types

from dau_sim.backends.cocotb_backend import (
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


def _make_dff():
    """D flip-flop: on posedge clk, q <= d.

    Equivalent to cocotb/examples/simple_dff/dff.sv.
    """
    return Module(
        name="dff",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("d", Shape(1)), PortDirection.INPUT),
            Port(Signal("q", Shape(1)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(Assign(target="q", value=SignalRef(shape=Shape(1), name="d")),),
            ),
        ),
    )


def _make_adder(width=4):
    """Combinational adder: X = A + B.

    Equivalent to cocotb/examples/adder/hdl/adder.sv (DATA_WIDTH=4).
    """
    return Module(
        name="adder",
        ports=(
            Port(Signal("A", Shape(width)), PortDirection.INPUT),
            Port(Signal("B", Shape(width)), PortDirection.INPUT),
            Port(Signal("X", Shape(width + 1)), PortDirection.OUTPUT),
        ),
        comb_blocks=(
            CombBlock(
                stmts=(
                    Assign(
                        target="X",
                        value=Binary(
                            shape=Shape(width + 1),
                            op=BinaryOp.ADD,
                            left=SignalRef(shape=Shape(width), name="A"),
                            right=SignalRef(shape=Shape(width), name="B"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _make_simple_counter():
    """8-bit counter with rst, ena, set, din.

    Equivalent to cocotb/examples/doc_examples/quickstart/simple_counter.sv::

        always_ff @(posedge clk) begin
            if (rst)          counter <= 0;
            else if (set)     counter <= din;
            else if (ena)     counter <= counter + 1;
            else              counter <= counter;
        end
    """
    w = 8
    return Module(
        name="simple_counter",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("rst", Shape(1)), PortDirection.INPUT),
            Port(Signal("ena", Shape(1)), PortDirection.INPUT),
            Port(Signal("set", Shape(1)), PortDirection.INPUT),
            Port(Signal("din", Shape(w)), PortDirection.INPUT),
            Port(Signal("counter", Shape(w)), PortDirection.OUTPUT),
        ),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE, rst="rst"),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    IfElse(
                        cond=SignalRef(shape=Shape(1), name="rst"),
                        then_body=(Assign(target="counter", value=Const(shape=Shape(w), value=0)),),
                        else_body=(
                            IfElse(
                                cond=SignalRef(shape=Shape(1), name="set"),
                                then_body=(
                                    Assign(
                                        target="counter",
                                        value=SignalRef(shape=Shape(w), name="din"),
                                    ),
                                ),
                                else_body=(
                                    IfElse(
                                        cond=SignalRef(shape=Shape(1), name="ena"),
                                        then_body=(
                                            Assign(
                                                target="counter",
                                                value=Binary(
                                                    shape=Shape(w),
                                                    op=BinaryOp.ADD,
                                                    left=SignalRef(shape=Shape(w), name="counter"),
                                                    right=Const(shape=Shape(w), value=1),
                                                ),
                                            ),
                                        ),
                                        else_body=(
                                            Assign(
                                                target="counter",
                                                value=SignalRef(shape=Shape(w), name="counter"),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


class CocotbExampleTestBase:
    """Mix-in that sets up / tears down a cocotb simulation environment."""

    def _setup(self, module):
        """Build engine, patch cocotb.simulator, return engine."""
        # isort: off
        import cocotb
        import cocotb.handle  # must precede cocotb._gpi_triggers (circular import)
        import cocotb._gpi_triggers
        import cocotb.simtime
        # isort: on

        engine = SimulationEngine(module)
        engine.build_handle_hierarchy()
        sim_module = _create_simulator_module(engine)

        # Save originals
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

        # Minimal regression manager stub for start_soon/create_task
        from cocotb._test import RunningTest
        from cocotb.regression import RegressionManager

        cocotb._regression_manager = RegressionManager.__new__(RegressionManager)
        cocotb._regression_manager._running_test = RunningTest.__new__(RunningTest)
        cocotb._regression_manager._running_test.tasks = []
        cocotb._regression_manager._running_test._main_task = None

        self._engine = engine
        return engine

    def _teardown(self):
        # isort: off
        import cocotb
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

    def _run_coroutine(self, coro, max_steps=10_000_000):
        """Schedule *coro*, start the write scheduler, and run the engine."""
        import cocotb
        from cocotb.task import Task

        task = Task(coro)
        cocotb._scheduler_inst._schedule_task_internal(task)
        cocotb.handle._start_write_scheduler()
        cocotb._scheduler_inst._event_loop()

        self._engine._running = True
        self._engine.run(max_steps=max_steps)

        cocotb.handle._stop_write_scheduler()


# Example 1: Simple DFF (cocotb/examples/simple_dff)


class TestDFF(CocotbExampleTestBase):
    """Port of cocotb/examples/simple_dff/test_dff.py::dff_simple_test."""

    def test_dff_d_propagates_to_q(self):
        """Test that d propagates to q on each rising clock edge."""
        import cocotb

        engine = self._setup(_make_dff())
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock

            results = []

            async def dff_test():
                dut = cocotb.top

                # Set initial input value to prevent floating
                dut.d.value = 0

                clock = Clock(dut.clk, 10, unit="us")
                clock.start(start_high=False)

                # Synchronize with the clock
                await RisingEdge(dut.clk)

                expected_val = 0  # Matches initial input value
                for i in range(10):
                    val = random.randint(0, 1)
                    dut.d.value = val
                    await RisingEdge(dut.clk)
                    actual = int(dut.q.value)
                    results.append((actual, expected_val))
                    expected_val = val

                # Check the final input on the next clock
                await RisingEdge(dut.clk)
                results.append((int(dut.q.value), expected_val))

                engine.stop()

            random.seed(42)  # deterministic
            self._run_coroutine(dff_test())

            # All 11 assertions should match (output lags input by one cycle)
            for i, (actual, expected) in enumerate(results):
                assert actual == expected, f"DFF output q was incorrect on cycle {i}: got {actual}, expected {expected}"
            assert len(results) == 11
        finally:
            self._teardown()

    def test_dff_holds_value_without_clock(self):
        """Verify q doesn't change when no clock edge occurs."""
        import cocotb

        engine = self._setup(_make_dff())
        try:
            from cocotb._gpi_triggers import RisingEdge, Timer
            from cocotb.clock import Clock

            async def hold_test():
                dut = cocotb.top
                dut.d.value = 1

                clock = Clock(dut.clk, 10, unit="us")
                clock.start(start_high=False)

                await RisingEdge(dut.clk)
                await RisingEdge(dut.clk)
                assert int(dut.q.value) == 1

                # Change d but don't wait for edge — q should still be 1
                dut.d.value = 0
                await Timer(1, unit="us")
                # q hasn't had a rising edge yet, should still be 1
                # (timer is shorter than half-clock period)
                assert int(dut.q.value) == 1

                engine.stop()

            self._run_coroutine(hold_test())
        finally:
            self._teardown()


# Example 2: Adder (cocotb/examples/adder)


def adder_model(a: int, b: int) -> int:
    """Reference model, same as cocotb/examples/adder/model/adder_model.py."""
    return a + b


class TestAdder(CocotbExampleTestBase):
    """Port of cocotb/examples/adder/tests/test_adder.py."""

    def test_adder_basic(self):
        """Test for 5 + 10 = 15."""
        import cocotb

        engine = self._setup(_make_adder())
        try:
            from cocotb._gpi_triggers import Timer

            async def adder_basic_test():
                dut = cocotb.top
                dut.A.value = 5
                dut.B.value = 10
                await Timer(2, unit="ns")
                assert int(dut.X.value) == adder_model(5, 10), f"Adder result is incorrect: {int(dut.X.value)} != 15"
                engine.stop()

            self._run_coroutine(adder_basic_test())
        finally:
            self._teardown()

    def test_adder_randomised(self):
        """Test for adding 2 random numbers multiple times."""
        import cocotb

        engine = self._setup(_make_adder())
        try:
            from cocotb._gpi_triggers import Timer

            async def adder_randomised_test():
                dut = cocotb.top
                for _ in range(10):
                    a = random.randint(0, 15)
                    b = random.randint(0, 15)
                    dut.A.value = a
                    dut.B.value = b
                    await Timer(2, unit="ns")
                    assert int(dut.X.value) == adder_model(a, b), (
                        f"Randomised test failed with: {int(dut.A.value)} + {int(dut.B.value)} = {int(dut.X.value)}"
                    )
                engine.stop()

            random.seed(100)
            self._run_coroutine(adder_randomised_test())
        finally:
            self._teardown()

    def test_adder_overflow(self):
        """Verify the 5-bit output captures the carry for max inputs."""
        import cocotb

        engine = self._setup(_make_adder())
        try:
            from cocotb._gpi_triggers import Timer

            async def overflow_test():
                dut = cocotb.top
                dut.A.value = 15
                dut.B.value = 15
                await Timer(2, unit="ns")
                assert int(dut.X.value) == 30
                engine.stop()

            self._run_coroutine(overflow_test())
        finally:
            self._teardown()


# Example 3: Simple Counter (cocotb/examples/doc_examples/quickstart)


class TestSimpleCounter(CocotbExampleTestBase):
    """Port of cocotb quickstart examples (simple_counter_testcases.py)."""

    def test_quickstart_1(self):
        """Quickstart 1: sequential enable/disable counting.

        Translated from cocotb quickstart_1:
        - Reset the counter
        - Enable for count_cycles rising edges
        - Disable and verify counter holds
        """
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import RisingEdge, Timer
            from cocotb.clock import Clock

            async def quickstart_1():
                dut = cocotb.top

                # Initial value
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0

                # Reset sequence and clock start
                dut.rst.value = 1
                input_clock = Clock(dut.clk, 10, unit="ns")
                input_clock.start()
                await Timer(5, "ns")

                # Re-synchronize with the clock
                await RisingEdge(dut.clk)
                dut.rst.value = 0

                # Activate ena
                dut.ena.value = 1

                count_cycles = 10
                for _ in range(count_cycles):
                    await RisingEdge(dut.clk)

                # Deactivate ena
                dut.ena.value = 0
                await RisingEdge(dut.clk)

                # Check that the counter output matches
                assert int(dut.counter.value) == count_cycles, f"Counter expected {count_cycles}, got {int(dut.counter.value)}"

                # Wait some time — counter should not increment when ena is 0
                await Timer(100, "ns")
                assert int(dut.counter.value) == count_cycles, f"Counter changed while disabled: {int(dut.counter.value)}"

                engine.stop()

            self._run_coroutine(quickstart_1())
        finally:
            self._teardown()

    def test_quickstart_2_enable_counter(self):
        """Quickstart 2 simplified: concurrent reset + enable coroutines.

        Demonstrates start_soon() and concurrent coroutines.
        """
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import FallingEdge, RisingEdge
            from cocotb.clock import Clock

            async def quickstart_2():
                dut = cocotb.top
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0

                # Start clock and reset
                dut.rst.value = 1
                clock = Clock(dut.clk, 10, unit="ns")
                clock.start()

                # Hold reset for several cycles
                for _ in range(3):
                    await RisingEdge(dut.clk)
                dut.rst.value = 0

                # Verify counter is 0 after reset
                await RisingEdge(dut.clk)
                assert int(dut.counter.value) == 0

                # Enable counting on falling edge
                await FallingEdge(dut.clk)
                dut.ena.value = 1

                # Count for 5 cycles
                for _ in range(5):
                    await RisingEdge(dut.clk)

                dut.ena.value = 0
                await RisingEdge(dut.clk)

                assert int(dut.counter.value) == 5, f"Expected 5, got {int(dut.counter.value)}"

                engine.stop()

            self._run_coroutine(quickstart_2())
        finally:
            self._teardown()

    def test_counter_set_din(self):
        """Verify the 'set' input loads din into counter."""
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock

            async def set_test():
                dut = cocotb.top
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0

                # Reset
                dut.rst.value = 1
                clock = Clock(dut.clk, 10, unit="ns")
                clock.start()
                await RisingEdge(dut.clk)
                dut.rst.value = 0

                # Load value via set
                await RisingEdge(dut.clk)
                dut.set.value = 1
                dut.din.value = 42
                await RisingEdge(dut.clk)
                await RisingEdge(dut.clk)
                dut.set.value = 0

                assert int(dut.counter.value) == 42, f"Expected counter=42 after set, got {int(dut.counter.value)}"

                # Enable counting from 42
                dut.ena.value = 1
                for _ in range(3):
                    await RisingEdge(dut.clk)
                dut.ena.value = 0
                await RisingEdge(dut.clk)

                assert int(dut.counter.value) == 45, f"Expected counter=45, got {int(dut.counter.value)}"

                engine.stop()

            self._run_coroutine(set_test())
        finally:
            self._teardown()

    def test_counter_reset_during_count(self):
        """Verify rst resets counter even while ena is active."""
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock

            async def reset_test():
                dut = cocotb.top
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0

                # Reset
                dut.rst.value = 1
                clock = Clock(dut.clk, 10, unit="ns")
                clock.start()
                await RisingEdge(dut.clk)
                dut.rst.value = 0

                # Count up
                await RisingEdge(dut.clk)
                dut.ena.value = 1
                for _ in range(5):
                    await RisingEdge(dut.clk)

                # NBA semantics: VCH fires before NBA, so counter
                # shows the pre-increment value (lags by one cycle).
                await RisingEdge(dut.clk)
                assert int(dut.counter.value) == 5

                # Reset while ena is still active
                dut.rst.value = 1
                await RisingEdge(dut.clk)
                await RisingEdge(dut.clk)

                assert int(dut.counter.value) == 0, f"Expected 0 after reset, got {int(dut.counter.value)}"

                # Release reset — counter should resume counting
                dut.rst.value = 0
                for _ in range(3):
                    await RisingEdge(dut.clk)
                dut.ena.value = 0
                await RisingEdge(dut.clk)

                assert int(dut.counter.value) == 3, f"Expected 3 after re-enabling, got {int(dut.counter.value)}"

                engine.stop()

            self._run_coroutine(reset_test())
        finally:
            self._teardown()

    def test_counter_start_soon(self):
        """Verify start_soon() works for concurrent clock + test coroutines."""
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import FallingEdge, RisingEdge
            from cocotb.clock import Clock

            results = []

            async def monitor(dut, n_cycles):
                """Continuously monitor counter value on each rising edge."""
                for _ in range(n_cycles):
                    await RisingEdge(dut.clk)
                    results.append(int(dut.counter.value))

            async def main_test():
                dut = cocotb.top
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0
                dut.rst.value = 1

                clock = Clock(dut.clk, 10, unit="ns")
                clock.start()
                await RisingEdge(dut.clk)
                dut.rst.value = 0
                await RisingEdge(dut.clk)

                # start_soon: launch monitor concurrently
                cocotb.start_soon(monitor(dut, 5))

                # Enable counter
                await FallingEdge(dut.clk)
                dut.ena.value = 1

                for _ in range(6):
                    await RisingEdge(dut.clk)

                engine.stop()

            self._run_coroutine(main_test())

            # Monitor should have recorded 5 counter values
            assert len(results) == 5
            # Values should be increasing
            for i in range(1, len(results)):
                assert results[i] >= results[i - 1]
        finally:
            self._teardown()

    def test_counter_holds_when_disabled(self):
        """Counter should hold its value when neither ena nor set is active."""
        import cocotb

        engine = self._setup(_make_simple_counter())
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock

            async def hold_test():
                dut = cocotb.top
                dut.ena.value = 0
                dut.set.value = 0
                dut.din.value = 0

                # Reset
                dut.rst.value = 1
                clock = Clock(dut.clk, 10, unit="ns")
                clock.start()
                await RisingEdge(dut.clk)
                dut.rst.value = 0

                # Count to 3
                await RisingEdge(dut.clk)
                dut.ena.value = 1
                for _ in range(3):
                    await RisingEdge(dut.clk)
                dut.ena.value = 0

                # Wait many more cycles — counter should stay at 3
                for _ in range(10):
                    await RisingEdge(dut.clk)

                assert int(dut.counter.value) == 3, f"Counter should hold at 3, got {int(dut.counter.value)}"

                engine.stop()

            self._run_coroutine(hold_test())
        finally:
            self._teardown()
