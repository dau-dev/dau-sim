"""Whether the Verilator on this machine can run the cocotb runner tests.

cocotb 2.1 compiles its own VPI shim against Verilator and needs 5.036 or
newer; Ubuntu 24.04's packaged 5.020 fails to compile it. The plain
Verilator integration tests have no such floor."""

from __future__ import annotations

import re
import subprocess
from shutil import which

import pytest

COCOTB_MIN_VERILATOR = (5, 36)


def verilator_version() -> tuple[int, int] | None:
    if which("verilator") is None:
        return None
    try:
        text = subprocess.run(("verilator", "--version"), capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Verilator (\d+)\.(\d+)", text)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def _reason() -> str:
    version = verilator_version()
    if version is None:
        return "verilator not found"
    return f"verilator {version[0]}.{version[1]:03d} is older than the {COCOTB_MIN_VERILATOR[0]}.{COCOTB_MIN_VERILATOR[1]:03d} cocotb needs"


requires_cocotb_verilator = pytest.mark.skipif((verilator_version() or (0, 0)) < COCOTB_MIN_VERILATOR, reason=_reason())
