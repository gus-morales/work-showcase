"""Renders the decision lifecycle as a diagram (not a chart, there's no
data behind it) to assets/lifecycle.png, reused as the portfolio
thumbnail for this project."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import set_style, INK, MUTED_TEAL, MUTED_RED, GREY, LIGHT_GREY, BG

BASE = Path(__file__).resolve().parents[1]
set_style()

MAIN = [
    ("draft", 1.0, MUTED_TEAL),
    ("approved", 3.4, MUTED_TEAL),
    ("shipped", 5.8, MUTED_TEAL),
    ("closed", 8.2, MUTED_TEAL),
]
Y_MAIN = 2.4
BOX_W, BOX_H = 1.7, 0.75


def draw_box(ax, x, y, label, color):
    box = FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.2, edgecolor=color, facecolor=LIGHT_GREY,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=12.5, color=INK, fontfamily="Lato")


def draw_arrow(ax, xy_from, xy_to, color=GREY, style="-", label=None, label_offset=(0, 0.18)):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, linestyle=style,
                         shrinkA=8, shrinkB=8),
    )
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="center",
                fontsize=9.5, color=color, fontfamily="Lato")


def main():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(-0.3, 9.7)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    for label, x, color in MAIN:
        draw_box(ax, x, Y_MAIN, label, color)
    for (l1, x1, _), (l2, x2, _) in zip(MAIN, MAIN[1:]):
        draw_arrow(ax, (x1 + BOX_W / 2, Y_MAIN), (x2 - BOX_W / 2, Y_MAIN))

    # abandoned: a draft that never gets approved
    draw_box(ax, 1.0, 0.6, "abandoned", MUTED_RED)
    draw_arrow(ax, (1.0, Y_MAIN - BOX_H / 2), (1.0, 0.6 + BOX_H / 2), color=MUTED_RED, style="--",
               label="never approved", label_offset=(1.05, 0))

    # reverted: a shipped decision that gets rolled back instead of closing out
    draw_box(ax, 8.2, 0.6, "reverted", MUTED_RED)
    draw_arrow(ax, (5.8, Y_MAIN - BOX_H / 2), (8.2, 0.6 + BOX_H / 2), color=MUTED_RED, style="--",
               label="rolled back", label_offset=(0, 0.35))

    ax.text(1.0, 3.85, "impact level shapes what's required at each step",
            ha="left", va="center", fontsize=13, color=INK, fontfamily="Lora")
    ax.text(1.0, 3.5, "low: outcome check only  ·  medium: + ship check  ·  high: + rollback plan, review board",
            ha="left", va="center", fontsize=10, color=GREY, fontfamily="Lato")

    fig.tight_layout()
    out_path = BASE / "assets" / "lifecycle.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
