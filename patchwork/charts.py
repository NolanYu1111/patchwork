"""Server-side charts rendered as inline SVG with Matplotlib."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#2b2620"
MUTED = "#8a8177"
GRID = "#e7e0d6"
LIGHT = "#c8743a"   # lighter-skin group
DARK = "#5b3a8c"    # darker-skin group
ALL = "#2f6f62"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.find("<svg"):]


def roc(curves: dict, contrast: list) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = [ALL, LIGHT, DARK]
    aurocs = {"All cases": None} | {g["group"]: g["auroc"] for g in contrast}
    for (name, (fpr, tpr)), c in zip(curves.items(), colors):
        label = name if aurocs.get(name) is None else f"{name} ({aurocs[name]:.3f})"
        ax.plot(fpr, tpr, color=c, lw=2.2 if name != "All cases" else 1.4, label=label,
                ls="-" if name != "All cases" else "--")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    return _svg(fig)


def forest(bands: list, contrast: list, split_at: int = 4) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs, labels = [], []
    for i, b in enumerate(bands):
        c = LIGHT if i < split_at else DARK
        if b.get("auroc") is not None:
            ax.errorbar(i, b["auroc"], yerr=[[b["auroc"] - b["ci_low"]], [b["ci_high"] - b["auroc"]]],
                        fmt="o", color=c, ms=6, capsize=3, lw=1.6)
        else:
            ax.text(i, 0.52, "n/a", ha="center", color=MUTED, fontsize=8)
        xs.append(i)
        labels.append(f"{b['group']}\nn={b['n']} · {b['positives']}+")
    for g, (lo, hi), c in zip(contrast, ((-0.4, split_at - 0.6), (split_at - 0.4, 5.4)), (LIGHT, DARK)):
        if g["auroc"] is not None:
            ax.hlines(g["auroc"], lo, hi, color=c, lw=5, alpha=0.25)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.axvline(split_at - 0.5, color=GRID, lw=1)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (95% CI)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    return _svg(fig)


def power_curve(curve: list, current_dark: int) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = [c["n_dark"] for c in curve]
    y = [c["power"] for c in curve]
    ax.plot(x, y, color=DARK, lw=2.2, marker="o", ms=5)
    ax.axhline(0.8, color=MUTED, ls="--", lw=1)
    ax.text(x[0], 0.82, "80% power", color=MUTED, fontsize=8)
    ax.axvline(current_dark, color=LIGHT, lw=1.2, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Darker-skin test cases (both groups scaled together)")
    ax.set_ylabel("Power to detect the gap")
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.6)
    return _svg(fig)
