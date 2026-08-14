"""A source may include a header that sits beside it.

Verilator does not search the including file's own directory, so
```include "types.svh"`` fails with "Cannot find include file" unless the
directory is passed as ``-I``. dau-core adding one shared header broke five
benches in dau this way, none of which had any reason to know a header
existed -- they just listed the tile sources they always listed.

The contract is that every directory contributing a source is an include
path. That is what a caller listing those sources already means, and it
keeps header adoption from being a change every downstream bench must make.

This checks the contract without a toolchain, so it runs everywhere. The
end-to-end proof -- a module that only compiles if the header is found --
needs verilator and lives in ``tests/integration/``.
"""

from __future__ import annotations

from pathlib import Path

from dau_sim.integrations.cocotb import run_cocotb_testbench


def test_every_source_directory_is_offered_to_the_build(tmp_path: Path, monkeypatch) -> None:
    """Stand in for the runner and check what the build was asked for."""
    import cocotb_tools.runner

    first, second = tmp_path / "a", tmp_path / "b"
    for directory in (first, second):
        directory.mkdir()
    sources = [first / "one.sv", second / "two.sv", first / "three.sv"]
    for source in sources:
        source.write_text("module m; endmodule\n")

    captured: dict[str, object] = {}

    class _Runner:
        def build(self, **kwargs) -> None:
            captured.update(kwargs)

        def test(self, **kwargs) -> None:
            captured["tested"] = True

    monkeypatch.setattr(cocotb_tools.runner, "get_runner", lambda _name: _Runner())
    run_cocotb_testbench(sources=sources, hdl_toplevel="m", test_module="x", build_dir=tmp_path / "build")

    assert captured["includes"] == [first, second], "one include per directory, deduplicated and ordered"
