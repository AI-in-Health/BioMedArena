"""Runtime bootstrap: auto-load .env once at package import time.

Any other module in this package that does `import harness` (or imports
from `harness.*`) gets the .env variables injected into os.environ before
any YAML `${VAR}` substitution runs.

No-op if python-dotenv isn't installed or if the .env file is missing —
we never raise.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_env_once(override: bool = False) -> None:
    """Idempotent: loads .env the first time, no-ops afterwards."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # python-dotenv missing — silent no-op

    # Search upward from this file for a .env at repo root
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent / ".env", Path.cwd() / ".env"]:
        if candidate.is_file():
            load_dotenv(candidate, override=override)
            return
