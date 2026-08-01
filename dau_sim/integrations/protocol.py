"""Runtime checkers for DAU valid/ready stream contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal


class StreamContractViolation(AssertionError):
    """A sampled interface violated its stream contract."""


class HandshakeContractChecker:
    """Simulator-neutral valid/ready stability checker."""

    def __init__(
        self,
        name: str,
        payload_names: Sequence[str],
        *,
        valid_rule: str = "VALID_HELD_UNTIL_READY",
        payload_rule: str = "PAYLOAD_STABLE_UNTIL_READY",
    ) -> None:
        if not name:
            raise ValueError("interface name must be non-empty")
        if not payload_names:
            raise ValueError("at least one payload signal is required")
        self.name = name
        self.payload_names = tuple(payload_names)
        self._valid_rule = valid_rule
        self._payload_rule = payload_rule
        self.cycle = 0
        self.transfers = 0
        self._held_payload: tuple[int, ...] | None = None

    def reset(self) -> None:
        self.cycle = 0
        self.transfers = 0
        self._held_payload = None

    def observe(self, *, valid: int, ready: int, payload: Mapping[str, int]) -> bool:
        """Sample one active clock edge and return whether a transfer fired."""
        self.cycle += 1
        valid_bit = self._binary_value("valid", valid)
        ready_bit = self._binary_value("ready", ready)
        values = tuple(int(payload[name]) for name in self.payload_names)

        if self._held_payload is not None:
            if not valid_bit:
                self._violate(self._valid_rule, "valid dropped before ready accepted the stalled transfer")
            if values != self._held_payload:
                changed = [
                    name for name, previous, current in zip(self.payload_names, self._held_payload, values, strict=True) if previous != current
                ]
                self._violate(self._payload_rule, f"stalled payload changed: {', '.join(changed)}")

        transfer = bool(valid_bit and ready_bit)
        if transfer:
            self.transfers += 1
            self._held_payload = None
        elif valid_bit:
            self._held_payload = values
        else:
            self._held_payload = None
        return transfer

    def finish(self) -> None:
        """Validate end-of-observation rules."""

    def _binary_value(self, signal: str, value: int) -> int:
        bit = int(value)
        if bit not in (0, 1):
            self._violate("BINARY_CONTROL", f"{signal} must be 0 or 1, got {bit}")
        return bit

    def _violate(self, rule: str, detail: str) -> None:
        raise StreamContractViolation(f"[{rule}] {self.name} cycle {self.cycle}: {detail}")


class StreamContractChecker(HandshakeContractChecker):
    """Check valid/ready stability and optional ``last`` framing."""

    def __init__(
        self,
        name: str,
        *,
        payload_names: Sequence[str] = ("data", "last"),
        last_name: str | None = "last",
        expected_batches: int | None = None,
    ) -> None:
        if expected_batches is not None and expected_batches < 0:
            raise ValueError("expected_batches must be non-negative")
        if last_name is not None and last_name not in payload_names:
            raise ValueError("last_name must be included in payload_names")
        super().__init__(name, payload_names)
        self.last_name = last_name
        self.expected_batches = expected_batches
        self.completed_batches = 0

    def reset(self) -> None:
        super().reset()
        self.completed_batches = 0

    def observe(self, *, valid: int, ready: int, payload: Mapping[str, int]) -> bool:
        transfer = super().observe(valid=valid, ready=ready, payload=payload)
        if not transfer or self.last_name is None:
            return transfer

        last = self._binary_value(self.last_name, payload[self.last_name])
        if self.expected_batches is not None and self.completed_batches >= self.expected_batches:
            if last:
                self._violate("LAST_EXACTLY_ONCE_PER_BATCH", "duplicate last transfer after all expected batches completed")
            self._violate("NO_TRANSFER_AFTER_LAST", "transfer occurred after the final batch's last")
        if last:
            self.completed_batches += 1
        return transfer

    def finish(self) -> None:
        if self.expected_batches is not None and self.completed_batches != self.expected_batches:
            self._violate(
                "LAST_EXACTLY_ONCE_PER_BATCH",
                f"missing last transfer: completed {self.completed_batches} of {self.expected_batches} expected batches",
            )


StatusMode = Literal["terminal", "mid_lane"]


class StatusContractChecker(HandshakeContractChecker):
    """Check status backpressure and terminal or mid-lane cardinality."""

    def __init__(
        self,
        name: str = "status_",
        *,
        mode: StatusMode,
        expected_batches: int | None = None,
        payload_names: Sequence[str] = ("error", "error_code"),
        error_name: str = "error",
    ) -> None:
        if mode not in ("terminal", "mid_lane"):
            raise ValueError("mode must be 'terminal' or 'mid_lane'")
        if expected_batches is not None and expected_batches < 0:
            raise ValueError("expected_batches must be non-negative")
        if error_name not in payload_names:
            raise ValueError("error_name must be included in payload_names")
        super().__init__(
            name,
            payload_names,
            valid_rule="STATUS_HELD_UNTIL_READY",
            payload_rule="STATUS_PAYLOAD_STABLE_UNTIL_READY",
        )
        self.mode = mode
        self.expected_batches = expected_batches
        self.error_name = error_name
        self.completed_statuses = 0

    def reset(self) -> None:
        super().reset()
        self.completed_statuses = 0

    def observe(self, *, valid: int, ready: int, payload: Mapping[str, int]) -> bool:
        transfer = super().observe(valid=valid, ready=ready, payload=payload)
        if not transfer:
            return False

        error = self._binary_value(self.error_name, payload[self.error_name])
        if self.mode == "mid_lane" and not error:
            self._violate("MID_LANE_STATUS_ONLY_ON_ERROR", "mid-lane emitted a success status")
        self.completed_statuses += 1
        if self.expected_batches is not None and self.completed_statuses > self.expected_batches:
            self._violate("STATUS_EXACTLY_ONCE_PER_BATCH", "more statuses than expected batches")
        return True

    def finish(self) -> None:
        if self.mode == "terminal" and self.expected_batches is not None and self.completed_statuses != self.expected_batches:
            self._violate(
                "STATUS_EXACTLY_ONCE_PER_BATCH",
                f"completed {self.completed_statuses} statuses for {self.expected_batches} expected batches",
            )


class _CocotbContractMonitor:
    def __init__(self, dut, clock, prefix: str, checker: HandshakeContractChecker, *, reset=None) -> None:
        self.checker = checker
        self.clock = clock
        self.reset = reset
        self._valid = self._resolve(dut, f"{prefix}valid")
        self._ready = self._resolve(dut, f"{prefix}ready")
        self._payload = {name: self._resolve(dut, f"{prefix}{name}") for name in checker.payload_names}
        self._task = None
        self._violation: StreamContractViolation | None = None

    async def __aenter__(self):
        import cocotb

        self._task = cocotb.start_soon(self._run())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        if self._task is not None and not self._task.done():
            self._task.kill()
        if exc_type is not None:
            return False
        if self._violation is not None:
            raise self._violation
        self.checker.finish()
        return False

    async def _run(self) -> None:
        from cocotb.triggers import RisingEdge

        while True:
            await RisingEdge(self.clock)
            if self.reset is not None and int(self.reset.value):
                self.checker.reset()
                continue
            try:
                self.checker.observe(
                    valid=int(self._valid.value),
                    ready=int(self._ready.value),
                    payload={name: int(handle.value) for name, handle in self._payload.items()},
                )
            except StreamContractViolation as exc:
                self._violation = exc
                return

    @staticmethod
    def _resolve(dut, name: str):
        try:
            return getattr(dut, name)
        except AttributeError as exc:
            raise ValueError(f"DUT has no contract signal {name!r}") from exc


class StreamContractMonitor(_CocotbContractMonitor):
    """Cocotb context manager attaching a checker to a prefixed stream."""

    def __init__(
        self,
        dut,
        clock,
        prefix: str,
        *,
        reset=None,
        payload_names: Sequence[str] = ("data", "last"),
        last_name: str | None = "last",
        expected_batches: int | None = None,
    ) -> None:
        checker = StreamContractChecker(
            prefix,
            payload_names=payload_names,
            last_name=last_name,
            expected_batches=expected_batches,
        )
        super().__init__(dut, clock, prefix, checker, reset=reset)


class StatusContractMonitor(_CocotbContractMonitor):
    """Cocotb context manager attaching a checker to a status interface."""

    def __init__(
        self,
        dut,
        clock,
        *,
        prefix: str = "status_",
        reset=None,
        mode: StatusMode,
        expected_batches: int | None = None,
        payload_names: Sequence[str] = ("error", "error_code"),
        error_name: str = "error",
    ) -> None:
        checker = StatusContractChecker(
            prefix,
            mode=mode,
            expected_batches=expected_batches,
            payload_names=payload_names,
            error_name=error_name,
        )
        super().__init__(dut, clock, prefix, checker, reset=reset)


__all__ = (
    "HandshakeContractChecker",
    "StatusContractChecker",
    "StatusContractMonitor",
    "StreamContractChecker",
    "StreamContractMonitor",
    "StreamContractViolation",
)
