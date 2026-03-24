from __future__ import annotations

from datetime import timedelta

from dau_sim.ir.types import Shape

__all__ = ("TestbenchContext", "TestbenchResult", "TestbenchTimeout")


class TestbenchTimeout(Exception):
    """Raised when a testbench exceeds the maximum cycle count."""


class TestbenchResult:
    """Result of a testbench execution."""

    def __init__(self, ctx: TestbenchContext):
        self.cycle = ctx.cycle
        self.signals = dict(ctx._signals)
        self.history = {k: list(v) for k, v in ctx._history.items()}
        self.passed = True
        self.error = None

    def __repr__(self):
        return f"TestbenchResult(cycle={self.cycle}, passed={self.passed})"


class TestbenchContext:
    """Interactive simulation context for testbench functions.

    Tracks signal values across simulation steps, supports setting inputs,
    reading outputs, and making assertions about signal values.
    """

    def __init__(
        self,
        compiled_module,
        clock_period: timedelta = timedelta(microseconds=1),
        clocks: dict[str, timedelta] | None = None,
        max_cycles: int = 10000,
    ):
        self._cm = compiled_module
        self._clock_period = clock_period
        self._clocks = clocks
        self._max_cycles = max_cycles
        self._cycle = 0
        self._signals: dict[str, int] = {}
        self._history: dict[str, list[int]] = {}
        self._shapes: dict[str, Shape] = {}

        # Initialize from module defaults
        for p in compiled_module.module.ports:
            self._signals[p.name] = p.signal.init
            self._shapes[p.name] = p.shape
        for s in compiled_module.module.signals:
            self._signals[s.name] = s.init
            self._shapes[s.name] = s.shape

        # Record cycle-0 snapshot
        for name, val in self._signals.items():
            self._history[name] = [val]

    @property
    def cycle(self) -> int:
        """Current simulation cycle."""
        return self._cycle

    def set(self, signal: str, value: int) -> None:
        """Set an input signal value. Takes effect on the next ``tick()``."""
        if signal not in self._signals:
            raise KeyError(f"Unknown signal: {signal!r}")
        self._signals[signal] = value

    def get(self, signal: str) -> int:
        """Get the current value of any signal."""
        if signal not in self._signals:
            raise KeyError(f"Unknown signal: {signal!r}")
        return self._signals[signal]

    def tick(self, n: int = 1) -> None:
        """Advance simulation by *n* clock cycles.

        Runs the compiled module for *n* cycles using the current signal
        values as initial state.  After execution, all signal values are
        updated to reflect the final simulated state.
        """
        if n < 1:
            raise ValueError("tick count must be >= 1")
        if self._cycle + n > self._max_cycles:
            raise TestbenchTimeout(f"Testbench exceeded max_cycles={self._max_cycles} (at cycle {self._cycle}, requested {n} more)")

        traces = self._cm.run(
            cycles=n,
            clock_period=self._clock_period,
            inputs=dict(self._signals),
            clocks=self._clocks,
        )

        # Update current values and record history
        for name in list(self._signals.keys()):
            trace = traces.get(name, [])
            if trace:
                self._signals[name] = trace[-1][1]
                # Each trace entry = one cycle's output
                for _, val in trace:
                    self._history[name].append(val)
            else:
                # Signal not emitted — repeat last value for each cycle
                last = self._signals[name]
                for _ in range(n):
                    self._history[name].append(last)

        self._cycle += n

    def assert_eq(self, signal: str, expected: int, msg: str = "") -> None:
        """Assert that *signal* currently equals *expected*."""
        actual = self.get(signal)
        if actual != expected:
            detail = f"Cycle {self._cycle}: {signal} = {actual}, expected {expected}"
            if msg:
                detail += f" ({msg})"
            raise AssertionError(detail)

    def assert_neq(self, signal: str, value: int, msg: str = "") -> None:
        """Assert that *signal* does not equal *value*."""
        actual = self.get(signal)
        if actual == value:
            detail = f"Cycle {self._cycle}: {signal} = {actual}, expected != {value}"
            if msg:
                detail += f" ({msg})"
            raise AssertionError(detail)

    def assert_stable(self, signal: str, cycles: int | None = None) -> None:
        """Assert that *signal* has been stable (unchanged).

        If *cycles* is given, checks that the last *cycles* recorded values
        are identical.  Otherwise checks all recorded history.
        """
        history = self._history.get(signal, [])
        if cycles is not None:
            if cycles < 1:
                raise ValueError("cycles must be >= 1")
            check = history[-cycles:]
        else:
            check = history

        if len(check) < 2:
            return  # Can't check stability with fewer than 2 samples

        first = check[0]
        for i, v in enumerate(check[1:], 1):
            if v != first:
                raise AssertionError(f"Signal {signal!r} changed at history index {i}: {first} → {v} (within last {len(check)} samples)")

    def assert_changed(self, signal: str, cycles: int | None = None) -> None:
        """Assert that *signal* changed at least once in recent history."""
        history = self._history.get(signal, [])
        if cycles is not None:
            check = history[-cycles:]
        else:
            check = history

        if len(check) < 2:
            return

        first = check[0]
        for v in check[1:]:
            if v != first:
                return
        raise AssertionError(f"Signal {signal!r} was stable at {first} (within last {len(check)} samples)")

    def history_of(self, signal: str) -> list[int]:
        """Return the full value history for a signal (one entry per cycle)."""
        if signal not in self._history:
            raise KeyError(f"Unknown signal: {signal!r}")
        return list(self._history[signal])
