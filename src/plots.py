"""Consistent matplotlib styling for every figure in the project."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HAM = "#2a78d6"     # blue
SPAM = "#eb6834"    # orange
ACCENT = "#1baf7a"  # aqua (third series)
GREY = "#b8b7b0"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
SERIES = [HAM, SPAM, ACCENT, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e2",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 110,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
