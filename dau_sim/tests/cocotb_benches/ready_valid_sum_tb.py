from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def sums_samples_and_holds_result_under_backpressure(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))

    dut.rst.value = 1
    dut.input_valid.value = 0
    dut.input_last.value = 0
    dut.input_value.value = 0
    dut.result_ready.value = 0
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.input_ready.value) == 1

    await _send_sample(dut, 7, last=False)
    await _send_sample(dut, -2, last=False)
    await _send_sample(dut, 11, last=True)

    await RisingEdge(dut.clk)
    assert int(dut.result_valid.value) == 1
    assert int(dut.result_value.value.to_signed()) == 16
    assert int(dut.input_ready.value) == 0

    dut.input_valid.value = 1
    dut.input_last.value = 1
    dut.input_value.value = 99
    await RisingEdge(dut.clk)
    assert int(dut.result_valid.value) == 1
    assert int(dut.result_value.value.to_signed()) == 16

    dut.input_valid.value = 0
    dut.input_last.value = 0
    dut.result_ready.value = 1
    await RisingEdge(dut.clk)
    dut.result_ready.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.result_valid.value) == 0
    assert int(dut.input_ready.value) == 1


async def _send_sample(dut, value: int, *, last: bool) -> None:
    while int(dut.input_ready.value) == 0:
        await RisingEdge(dut.clk)
    dut.input_value.value = value
    dut.input_last.value = int(last)
    dut.input_valid.value = 1
    await RisingEdge(dut.clk)
    dut.input_valid.value = 0
    dut.input_last.value = 0
