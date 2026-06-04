from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import which


class VerilatorUnavailableError(RuntimeError):
    pass


class VerilatorExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerilatorTestbenchResult:
    compile_command: tuple[str, ...]
    run_command: tuple[str, ...]
    executable_path: Path
    stdout: str
    stderr: str
    compile_stdout: str
    compile_stderr: str


def run_verilator_testbench(
    *,
    sources: Sequence[Path | str],
    top_module: str,
    work_dir: Path | str,
    verilator: str = "verilator",
    extra_args: Sequence[str] = (),
) -> VerilatorTestbenchResult:
    verilator_path = which(verilator)
    if verilator_path is None:
        raise VerilatorUnavailableError(f"verilator executable not found: {verilator}")
    if not top_module:
        raise ValueError("top_module must be non-empty")

    source_paths = tuple(Path(source) for source in sources)
    if not source_paths:
        raise ValueError("at least one SystemVerilog source is required")
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    root = Path(work_dir)
    obj_dir = root / "obj_dir"
    obj_dir.mkdir(parents=True, exist_ok=True)

    compile_command = (
        verilator_path,
        "--binary",
        "--timing",
        "-Wno-fatal",
        "-Wno-DECLFILENAME",
        "-Mdir",
        str(obj_dir),
        "--top-module",
        top_module,
        *extra_args,
        *(str(source_path) for source_path in source_paths),
    )
    compile_result = subprocess.run(compile_command, cwd=root, capture_output=True, text=True, check=False)
    if compile_result.returncode != 0:
        raise VerilatorExecutionError(_format_failure("verilator compile", compile_command, compile_result))

    executable_path = obj_dir / f"V{top_module}"
    run_command = (str(executable_path),)
    run_result = subprocess.run(run_command, cwd=root, capture_output=True, text=True, check=False)
    if run_result.returncode != 0:
        raise VerilatorExecutionError(_format_failure("verilator run", run_command, run_result))

    return VerilatorTestbenchResult(
        compile_command=compile_command,
        run_command=run_command,
        executable_path=executable_path,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        compile_stdout=compile_result.stdout,
        compile_stderr=compile_result.stderr,
    )


def _format_failure(label: str, command: Sequence[str], result: subprocess.CompletedProcess[str]) -> str:
    command_text = " ".join(command)
    return f"{label} failed with exit code {result.returncode}: {command_text}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
