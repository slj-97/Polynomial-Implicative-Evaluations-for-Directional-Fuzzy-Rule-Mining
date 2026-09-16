"""Shared bootstrap helpers for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_clean_real_outputs(paths: list[Path]) -> None:
    """Guard against treating absent real-data outputs as evidence."""
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required measured input is missing: " + ", ".join(missing))
