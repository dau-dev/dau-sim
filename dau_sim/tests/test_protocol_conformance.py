from __future__ import annotations

import pytest

from dau_sim.compiler import compile_module
from dau_sim.integrations.protocol import StatusContractChecker, StreamContractChecker, StreamContractMonitor, StreamContractViolation
from dau_sim.ir import Assign, Binary, BinaryOp, ClockDomain, Const, EdgePolarity, Module, Port, PortDirection, SeqBlock, Shape, Signal, SignalRef
from dau_sim.tests.test_cocotb_examples import CocotbExampleTestBase


def _broken_stream_module(failure: str) -> Module:
    count = SignalRef(shape=Shape(4), name="count")
    valid = Const(shape=Shape(1), value=1)
    data = Const(shape=Shape(8), value=17)
    last = Const(shape=Shape(1), value=0)

    if failure == "drops_valid":
        valid = Binary(shape=Shape(1), op=BinaryOp.EQ, left=count, right=Const(shape=Shape(4), value=0))
    elif failure == "mutates_payload":
        data = Binary(shape=Shape(8), op=BinaryOp.ADD, left=SignalRef(shape=Shape(8), name="output_data"), right=Const(shape=Shape(8), value=1))
    elif failure == "emits_two_lasts":
        last = Const(shape=Shape(1), value=1)
    elif failure != "never_emits_last":
        raise ValueError(f"unknown failure {failure!r}")

    return Module(
        name=f"broken_stream_{failure}",
        ports=(
            Port(Signal("clk", Shape(1)), PortDirection.INPUT),
            Port(Signal("output_ready", Shape(1)), PortDirection.INPUT),
            Port(Signal("output_valid", Shape(1)), PortDirection.OUTPUT),
            Port(Signal("output_data", Shape(8)), PortDirection.OUTPUT),
            Port(Signal("output_last", Shape(1)), PortDirection.OUTPUT),
        ),
        signals=(Signal("count", Shape(4)),),
        clock_domains=(ClockDomain("sync", clk="clk", edge=EdgePolarity.POSEDGE),),
        seq_blocks=(
            SeqBlock(
                domain="sync",
                stmts=(
                    Assign("output_valid", valid),
                    Assign("output_data", data),
                    Assign("output_last", last),
                    Assign("count", Binary(shape=Shape(4), op=BinaryOp.ADD, left=count, right=Const(shape=Shape(4), value=1))),
                ),
            ),
        ),
    )


def _check_broken_module(failure: str, *, ready: int) -> None:
    module = _broken_stream_module(failure)
    traces = compile_module(module).run(cycles=4, inputs={"output_ready": ready})
    checker = StreamContractChecker("output_", expected_batches=1)
    for index in range(len(traces["clk"])):
        checker.observe(
            valid=traces["output_valid"][index][1],
            ready=traces["output_ready"][index][1],
            payload={"data": traces["output_data"][index][1], "last": traces["output_last"][index][1]},
        )
    checker.finish()


def test_broken_module_that_drops_valid_identifies_hold_rule() -> None:
    with pytest.raises(StreamContractViolation, match=r"\[VALID_HELD_UNTIL_READY\] output_ cycle 2"):
        _check_broken_module("drops_valid", ready=0)


def test_broken_module_that_mutates_payload_identifies_stability_rule() -> None:
    with pytest.raises(StreamContractViolation, match=r"\[PAYLOAD_STABLE_UNTIL_READY\] output_ cycle 2"):
        _check_broken_module("mutates_payload", ready=0)


def test_broken_module_that_emits_two_lasts_identifies_last_rule() -> None:
    with pytest.raises(StreamContractViolation, match=r"\[LAST_EXACTLY_ONCE_PER_BATCH\] output_ cycle 2"):
        _check_broken_module("emits_two_lasts", ready=1)


def test_broken_module_that_never_emits_last_identifies_last_rule() -> None:
    with pytest.raises(StreamContractViolation, match=r"\[LAST_EXACTLY_ONCE_PER_BATCH\] output_ cycle 4"):
        _check_broken_module("never_emits_last", ready=1)


def test_transfer_count_only_advances_when_valid_and_ready_are_high() -> None:
    checker = StreamContractChecker("input_", expected_batches=None)
    for valid, ready in ((0, 0), (0, 1), (1, 0), (1, 1)):
        checker.observe(valid=valid, ready=ready, payload={"data": 9, "last": 0})
    assert checker.transfers == 1


def test_transfer_after_final_last_is_rejected() -> None:
    checker = StreamContractChecker("output_", expected_batches=1)
    checker.observe(valid=1, ready=1, payload={"data": 9, "last": 1})
    with pytest.raises(StreamContractViolation, match="NO_TRANSFER_AFTER_LAST"):
        checker.observe(valid=1, ready=1, payload={"data": 10, "last": 0})


def test_terminal_status_requires_one_held_status_per_expected_batch() -> None:
    checker = StatusContractChecker(mode="terminal", expected_batches=1)
    checker.observe(valid=1, ready=0, payload={"error": 0, "error_code": 0})
    checker.observe(valid=1, ready=1, payload={"error": 0, "error_code": 0})
    checker.finish()
    assert checker.completed_statuses == 1


def test_status_must_be_held_until_ready() -> None:
    checker = StatusContractChecker(mode="terminal", expected_batches=1)
    checker.observe(valid=1, ready=0, payload={"error": 0, "error_code": 0})
    with pytest.raises(StreamContractViolation, match="STATUS_HELD_UNTIL_READY"):
        checker.observe(valid=0, ready=0, payload={"error": 0, "error_code": 0})


def test_mid_lane_rejects_success_status() -> None:
    checker = StatusContractChecker(mode="mid_lane", expected_batches=1)
    with pytest.raises(StreamContractViolation, match="MID_LANE_STATUS_ONLY_ON_ERROR"):
        checker.observe(valid=1, ready=1, payload={"error": 0, "error_code": 0})


class TestCocotbStreamContractMonitor(CocotbExampleTestBase):
    def test_context_manager_attaches_to_prefixed_interface(self) -> None:
        import cocotb

        engine = self._setup(_broken_stream_module("mutates_payload"))
        violations: list[str] = []
        try:
            from cocotb._gpi_triggers import RisingEdge
            from cocotb.clock import Clock

            async def run_bench() -> None:
                dut = cocotb.top
                dut.output_ready.value = 0
                Clock(dut.clk, 10, unit="ns").start(start_high=False)
                try:
                    async with StreamContractMonitor(dut, dut.clk, "output_", expected_batches=1):
                        for _ in range(4):
                            await RisingEdge(dut.clk)
                except StreamContractViolation as exc:
                    violations.append(str(exc))
                engine.stop()

            self._run_coroutine(run_bench())
        finally:
            self._teardown()

        assert len(violations) == 1
        assert "[PAYLOAD_STABLE_UNTIL_READY] output_ cycle 3" in violations[0]
