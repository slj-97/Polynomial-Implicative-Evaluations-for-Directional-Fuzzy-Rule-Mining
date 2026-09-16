"""LaTeX table export from measured result frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_latex_table(frame: pd.DataFrame, path: str | Path, *, caption: str, label: str) -> Path:
    if frame.empty:
        raise ValueError("Cannot export an empty result table as a manuscript table.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latex = frame.to_latex(index=False, escape=True, caption=caption, label=label, float_format="%.6g")
    output.write_text(latex, encoding="utf-8")
    return output
