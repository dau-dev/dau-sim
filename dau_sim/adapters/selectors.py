"""Signal selectors: glob/regex patterns for filtering simulation traces.

Use signal selectors to choose which signals to include in VCD output
or testbench observation, avoiding the overhead of tracing everything
in large designs.

Usage::

    from dau_sim.adapters.selectors import select_signals

    # Glob patterns
    filtered = select_signals(traces, include=["count", "en"])
    filtered = select_signals(traces, include=["*clk*"])
    filtered = select_signals(traces, exclude=["rst", "*internal*"])

    # Regex patterns
    filtered = select_signals(traces, include=[r"data_\\d+"], regex=True)

    # With CompiledModule
    traces = cm.run(cycles=100, inputs={"en": 1})
    cm.write_vcd("out.vcd", traces, signals=["count", "en"])
"""

from __future__ import annotations

import fnmatch
import re

__all__ = ["select_signals", "match_signals"]


def match_signals(
    names: list[str] | set[str],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    regex: bool = False,
) -> list[str]:
    """Return signal names matching the include/exclude filters.

    Parameters
    ----------
    names
        All available signal names.
    include
        Patterns to include.  If ``None``, all names are included.
    exclude
        Patterns to exclude.  Applied after include filtering.
    regex
        If ``True``, patterns are Python regexes.
        If ``False`` (default), patterns are Unix glob patterns.

    Returns
    -------
    list[str]
        Matching signal names, in the order they appear in *names*.
    """
    result = list(names)

    if include is not None:
        result = _apply_patterns(result, include, regex, mode="include")

    if exclude is not None:
        excluded = set(_apply_patterns(result, exclude, regex, mode="include"))
        result = [n for n in result if n not in excluded]

    return result


def select_signals(
    traces: dict[str, list],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    regex: bool = False,
) -> dict[str, list]:
    """Filter a traces dict to only include matching signals.

    Parameters
    ----------
    traces
        Signal name → trace list mapping (from ``CompiledModule.run()``).
    include
        Glob or regex patterns for signals to keep.
    exclude
        Glob or regex patterns for signals to remove.
    regex
        Interpret patterns as regex (default: glob).

    Returns
    -------
    dict[str, list]
        Filtered traces dict.
    """
    matched = match_signals(
        list(traces.keys()),
        include=include,
        exclude=exclude,
        regex=regex,
    )
    return {name: traces[name] for name in matched}


def _apply_patterns(
    names: list[str],
    patterns: list[str],
    regex: bool,
    mode: str,
) -> list[str]:
    """Return names that match any of the given patterns."""
    matched = []
    seen = set()

    for pattern in patterns:
        if regex:
            compiled = re.compile(pattern)
            for name in names:
                if name not in seen and compiled.search(name):
                    matched.append(name)
                    seen.add(name)
        else:
            for name in names:
                if name not in seen and fnmatch.fnmatch(name, pattern):
                    matched.append(name)
                    seen.add(name)

    return matched
