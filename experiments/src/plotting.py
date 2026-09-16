"""Publication artifact helpers; figures are created only from measured tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line_figure(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    if frame.empty:
        raise ValueError("Cannot create a figure from an empty result table.")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, subset in frame.groupby(group, dropna=False):
        ordered = subset.sort_values(x)
        axis.plot(ordered[x], ordered[y], marker="o", label=str(name))
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf)
    figure.savefig(png, dpi=600)
    plt.close(figure)
    return pdf, png
