"""Plot style and small helpers for the results notebook (notebooks/results.ipynb).

Follows the dataviz method: one fixed hue per entity (never cycled), thin marks, recessive grid, direct labels, text in ink colours
(never in the series colour). The three-hue palette was validated with the skill's validate_palette script (light surface,
all-pairs colour-blind separation >= 9). The aqua slot is below 3:1 contrast on the surface, so every aqua mark is labelled directly.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# colour follows the entity: the base model is grey (context), each adapter has its own hue in every figure
COND = {"base": MUTED, "bad_medical_advice": BLUE, "extreme_sports": ORANGE}
LABEL = {"base": "base model", "bad_medical_advice": "medical adapter", "extreme_sports": "sports adapter"}
FIG_DIR = "outputs/figures"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "regular", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "lines.linewidth": 2, "lines.markersize": 8,
    })


def bootstrap_mean(values, resamples=4000, seed=0):
    """(mean, 2.5th, 97.5th percentile) of per-question values, resampling whole questions."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), (resamples, len(v)))
    boot = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def dot_ci(ax, x, mean, lo, hi, color, label=None, fmt="{:.1f}%", scale=100, where="above", size=8, dy=0):
    """A dot with a thin interval; the value is printed in ink as a direct label: above the interval, or beside the dot (where='left'/'right')."""
    ax.plot([x, x], [lo * scale, hi * scale], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([x], [mean * scale], "o", color=color, ms=size, mec=SURFACE, mew=2, zorder=3)
    if label is not None:
        text = label if isinstance(label, str) else fmt.format(mean * scale)
        if where == "above":
            ax.annotate(text, (x, hi * scale), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=10)
        else:
            sign = -1 if where == "left" else 1
            ax.annotate(text, (x, mean * scale), xytext=(sign * 9, dy), textcoords="offset points", ha="right" if sign < 0 else "left", va="center", color=INK, fontsize=10)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=160, bbox_inches="tight")
