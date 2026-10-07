"""The README's python block runs as written."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def test_the_readme_example_runs_as_written(tmp_path, monkeypatch) -> None:
    blocks = BLOCK.findall(README.read_text(encoding="utf-8"))
    assert blocks, "the README has no python block to run"
    monkeypatch.chdir(tmp_path)
    # a real module, so classes defined in the block resolve their annotations
    # the way they would in a script (amaranth reads them through sys.modules)
    module = types.ModuleType("readme_example")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"README.md block {index + 1}", "exec", dont_inherit=True), module.__dict__)  # noqa: S102
        except Exception as error:
            raise AssertionError(f"README python block {index + 1} does not run as written: {error!r}\n{block}") from error
