from __future__ import annotations

import threading
from datetime import timedelta

from dau_sim.ir.types import Shape

__all__ = ("TestbenchContext", "TestbenchResult", "TestbenchTimeout", "run_parallel_testbenches")


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

        # Signal forcing state
        self._forced: dict[str, int] = {}  # signal_name -> forced value

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

    def force(self, signal: str, value: int) -> None:
        """Force a signal to a constant value, overriding simulation output.

        The forced value is applied after each ``tick()`` and persists
        until ``release()`` is called.
        """
        if signal not in self._signals:
            raise KeyError(f"Unknown signal: {signal!r}")
        self._forced[signal] = value
        self._signals[signal] = value

    def release(self, signal: str) -> None:
        """Release a previously forced signal, restoring simulation control."""
        if signal not in self._forced:
            raise KeyError(f"Signal {signal!r} is not forced")
        del self._forced[signal]

    def is_forced(self, signal: str) -> bool:
        """Return True if the signal is currently forced."""
        return signal in self._forced

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

        # Apply forced signal overrides
        for sig, val in self._forced.items():
            self._signals[sig] = val

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


class _ParallelView:
    """Per-thread view of a shared :class:`TestbenchContext`.

    Each parallel testbench function receives one of these.  Signal reads
    and writes go through a shared lock, and ``tick()`` uses a
    :class:`threading.Barrier` so all threads advance together.
    """

    def __init__(self, ctx: TestbenchContext, barrier: threading.Barrier, lock: threading.Lock, index: int):
        self._ctx = ctx
        self._barrier = barrier
        self._lock = lock
        self._index = index
        self._error: Exception | None = None

    @property
    def cycle(self) -> int:
        return self._ctx.cycle

    def set(self, signal: str, value: int) -> None:
        with self._lock:
            self._ctx.set(signal, value)

    def get(self, signal: str) -> int:
        with self._lock:
            return self._ctx.get(signal)

    def force(self, signal: str, value: int) -> None:
        with self._lock:
            self._ctx.force(signal, value)

    def release(self, signal: str) -> None:
        with self._lock:
            self._ctx.release(signal)

    def is_forced(self, signal: str) -> bool:
        with self._lock:
            return self._ctx.is_forced(signal)

    def tick(self, n: int = 1) -> None:
        """Synchronize with all parallel testbenches, then advance *n* cycles.

        Thread 0 (the first to arrive) performs the actual simulation tick.
        All other threads wait at the barrier.
        """
        for _ in range(n):
            # Phase 1: all threads arrive at the barrier
            idx = self._barrier.wait()

            # Phase 2: exactly one thread (the one that gets index 0 from barrier)
            # performs the tick
            if idx == 0:
                self._ctx.tick(1)

            # Phase 3: wait until tick is complete before any thread proceeds
            self._barrier.wait()

    def assert_eq(self, signal: str, expected: int, msg: str = "") -> None:
        with self._lock:
            self._ctx.assert_eq(signal, expected, msg)

    def assert_neq(self, signal: str, value: int, msg: str = "") -> None:
        with self._lock:
            self._ctx.assert_neq(signal, value, msg)

    def assert_stable(self, signal: str, cycles: int | None = None) -> None:
        with self._lock:
            self._ctx.assert_stable(signal, cycles)

    def assert_changed(self, signal: str, cycles: int | None = None) -> None:
        with self._lock:
            self._ctx.assert_changed(signal, cycles)

    def history_of(self, signal: str) -> list[int]:
        with self._lock:
            return self._ctx.history_of(signal)


def run_parallel_testbenches(
    compiled_module,
    *fns,
    clock_period: timedelta = timedelta(microseconds=1),
    clocks: dict[str, timedelta] | None = None,
    max_cycles: int = 10000,
) -> TestbenchResult:
    """Run multiple testbench functions in parallel against a compiled module.

    Each function receives a :class:`_ParallelView` that supports the same
    API as :class:`TestbenchContext`.  All threads synchronize at each
    ``tick()`` call so the simulation advances in lock-step.

    Returns a :class:`TestbenchResult` capturing the final state.

    Raises the first exception from any thread if a testbench fails.
    """
    if not fns:
        raise ValueError("At least one testbench function is required")

    ctx = TestbenchContext(compiled_module, clock_period, clocks, max_cycles)
    barrier = threading.Barrier(len(fns))
    lock = threading.Lock()

    views = [_ParallelView(ctx, barrier, lock, i) for i in range(len(fns))]
    errors: list[tuple[int, Exception]] = []
    errors_lock = threading.Lock()

    def _worker(fn, view):
        try:
            fn(view)
        except Exception as e:
            with errors_lock:
                errors.append((view._index, e))
            # Abort the barrier so other threads don't hang forever
            barrier.abort()

    threads = [threading.Thread(target=_worker, args=(fn, view), daemon=True) for fn, view in zip(fns, views)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    result = TestbenchResult(ctx)
    if errors:
        errors.sort(key=lambda x: x[0])
        result.passed = False
        result.error = errors[0][1]
        raise errors[0][1]
    result.passed = True
    return result
