"""Public tool registry exports.

The canonical TOOL_SPECS list lives in ``function_calling_runner``. Keep this
module tiny so imports such as ``from harness.tools import TOOL_SPECS`` remain
stable without eagerly importing optional tool backends unless requested.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "TOOL_SPECS":
        from harness.eval.function_calling_runner import TOOL_SPECS

        return TOOL_SPECS
    raise AttributeError(name)


__all__ = ["TOOL_SPECS"]
