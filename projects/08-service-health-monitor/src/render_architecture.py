"""Renders the snapshot-pipeline architecture as a diagram (not a chart,
there's no data behind it) to assets/architecture.png, reused as the
portfolio thumbnail for this project."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import set_style, INK, MUTED_TEAL, MUTED_AMBER, GREY, LIGHT_GREY, BG

BASE = Path(__file__).resolve().parents[1]
set_style()

BOX_W, BOX_H = 2.5, 0.85


def draw_box(ax, x, y, label, color, w=BOX_W, h=BOX_H, fontsize=11.5):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        linewidth=1.2, edgecolor=color, facecolor=LIGHT_GREY,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=INK,
             fontfamily="Lato", linespacing=1.4)


def draw_arrow(ax, xy_from, xy_to, color=GREY, style="-", label=None, label_offset=(0, 0)):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, linestyle=style,
                         shrinkA=6, shrinkB=6),
    )
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="center",
                fontsize=9, color=color, fontfamily="Lato")


def main():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0.5, 5.3)
    ax.axis("off")

    draw_box(ax, 2.0, 4.1, "Metrics panel\n(data/service_metrics.csv)", MUTED_TEAL, w=2.7)
    draw_box(ax, 5.9, 4.1, "Detector engine\nthreshold · zscore · trend_break · data_gap", MUTED_TEAL, w=3.7)
    draw_box(ax, 9.9, 4.1, "Snapshot\nflags.csv + events.json", MUTED_AMBER, w=2.8)

    draw_arrow(ax, (2.0 + 2.7 / 2, 4.1), (5.9 - 3.7 / 2, 4.1))
    draw_arrow(ax, (5.9 + 3.7 / 2, 4.1), (9.9 - 2.8 / 2, 4.1))

    draw_box(ax, 6.9, 1.9, "Alert channels\nconsole · file · webhook (dedup'd)", MUTED_TEAL, w=3.4)
    draw_box(ax, 10.5, 1.9, "Notebook / report\n(reads the snapshot only)", MUTED_TEAL, w=2.9)

    draw_arrow(ax, (9.2, 4.1 - BOX_H / 2), (7.5, 1.9 + BOX_H / 2))
    draw_arrow(ax, (9.9, 4.1 - BOX_H / 2), (10.5, 1.9 + BOX_H / 2))

    ax.text(0.2, 5.05, "Compute the snapshot once, everything downstream just reads it",
            ha="left", va="center", fontsize=13.5, color=INK, fontfamily="Lora")
    ax.text(0.2, 0.85,
            "Neither the alert channels nor the notebook ever recompute a detector or touch\n"
            "the metrics panel directly; a slow or broken data source can't take them down.",
            ha="left", va="center", fontsize=10, color=GREY, fontfamily="Lato")

    fig.tight_layout()
    out_path = BASE / "assets" / "architecture.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
