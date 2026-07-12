"""Canonical cocotb-on-Verilator launcher.

One flag set for every functional cocotb bench: consumers build and run
through this function instead of each carrying its own
``cocotb_tools.runner`` invocation, so compile behavior cannot drift
between test suites (the same rule the SV benches follow through
``run_verilator_testbench``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_BUILD_ARGS = ("--timing", "-Wno-fatal")


class CocotbProfile(BaseModel):
    """A registered cocotb bench: HDL sources, the toplevel they build,
    and the cocotb test module that drives it."""

    model_config = ConfigDict(frozen=True)

    name: str
    sources: tuple[Path, ...]
    hdl_toplevel: str
    test_module: str


class CocotbRunnerUnavailableError(RuntimeError):
    pass


def run_cocotb_testbench(
    *,
    sources: Sequence[Path | str],
    hdl_toplevel: str,
    test_module: str,
    build_dir: Path | str,
    results_xml: Path | str | None = None,
    build_args: Sequence[str] = DEFAULT_BUILD_ARGS,
    parameters: Mapping[str, object] | None = None,
    always: bool = True,
) -> Path:
    """Build the sources with Verilator and run the cocotb test module
    against ``hdl_toplevel``. Returns the results XML path; raises on any
    failing test (the cocotb runner's behavior)."""
    try:
        from cocotb_tools.runner import get_runner
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise CocotbRunnerUnavailableError("cocotb runner not installed (pip install cocotb)") from exc

    if not hdl_toplevel:
        raise ValueError("hdl_toplevel must be non-empty")
    source_paths = tuple(Path(source) for source in sources)
    if not source_paths:
        raise ValueError("at least one HDL source is required")
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    build_root = Path(build_dir)
    results_path = Path(results_xml) if results_xml is not None else build_root.parent / f"{hdl_toplevel}-cocotb-results.xml"

    runner = get_runner("verilator")
    build_kwargs = {"parameters": dict(parameters)} if parameters else {}
    runner.build(
        sources=source_paths,
        hdl_toplevel=hdl_toplevel,
        build_dir=build_root,
        always=always,
        build_args=tuple(build_args),
        **build_kwargs,
    )
    runner.test(
        hdl_toplevel=hdl_toplevel,
        test_module=test_module,
        build_dir=build_root,
        results_xml=str(results_path),
    )
    return results_path
