"""Renders the two-engine snapshot-pipeline architecture as a diagram
(not a chart, there's no data behind it) to assets/architecture.png,
reused as the portfolio thumbnail for this project."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import set_style, BG, GREY, INK, LIGHT_GREY, MUTED_AMBER, MUTED_TEAL

BASE = Path(__file__).resolve().parents[1]
set_style()

BOX_H = 0.85


def draw_box(ax, x, y, label, color, w=2.5, h=BOX_H, fontsize=10.8):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        linewidth=1.2, edgecolor=color, facecolor=LIGHT_GREY,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=INK,
             fontfamily="Lato", linespacing=1.4)
    return w, h


def draw_arrow(ax, xy_from, xy_to, color=GREY, style="-", label=None, label_offset=(0, 0)):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, linestyle=style,
                         shrinkA=6, shrinkB=6),
    )
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="center",
                fontsize=8.7, color=color, fontfamily="Lato")


def main():
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0.4, 6.7)
    ax.axis("off")

    # source + config
    db_w, _ = draw_box(ax, 2.0, 5.5, "DuckDB\npipeline_runs + scoring_log", MUTED_TEAL, w=3.0)
    cat_w, cat_h = draw_box(ax, 2.0, 3.9, "Metric catalog\ncatalog/*.yaml", MUTED_AMBER, w=3.0, h=0.7,
                             fontsize=10)

    # two engines
    ops_w, _ = draw_box(ax, 6.4, 6.2, "Ops detector engine\nthreshold · zscore · trend_break · data_gap",
                         MUTED_TEAL, w=3.7)
    pop_w, _ = draw_box(ax, 6.4, 4.7, "popmon stability engine\ndistribution drift, feature by feature",
                         MUTED_TEAL, w=3.7)

    draw_arrow(ax, (2.0 + db_w / 2, 5.5), (6.4 - ops_w / 2 - 0.1, 6.1), label="ops metrics", label_offset=(-0.3, 0.3))
    draw_arrow(ax, (2.0 + db_w / 2, 5.3), (6.4 - pop_w / 2 - 0.1, 4.8), label="model features", label_offset=(-0.2, -0.35))
    draw_arrow(ax, (2.0 + cat_w / 2, 3.9), (6.4 - ops_w / 2, 5.6), color=MUTED_AMBER, style="--")
    draw_arrow(ax, (2.0 + cat_w / 2, 3.9), (6.4 - pop_w / 2, 4.6), color=MUTED_AMBER, style="--",
               label="thresholds / dtypes", label_offset=(0.9, -1.1))

    # unified snapshot
    snap_w, _ = draw_box(ax, 10.6, 5.45, "Snapshot\nflags.csv + events.json", MUTED_AMBER, w=2.9)
    draw_arrow(ax, (6.4 + ops_w / 2, 6.0), (10.6 - snap_w / 2, 5.65))
    draw_arrow(ax, (6.4 + pop_w / 2, 4.9), (10.6 - snap_w / 2, 5.25))

    # downstream consumers, all snapshot-only
    draw_box(ax, 4.3, 1.6, "Alert channels\nconsole · file · webhook (dedup'd)", MUTED_TEAL, w=3.2)
    draw_box(ax, 7.7, 1.6, "Static dashboard\n+ notebook", MUTED_TEAL, w=2.9)
    draw_box(ax, 10.9, 1.6, "popmon HTML report\nfull per-feature detail", MUTED_TEAL, w=3.0)

    draw_arrow(ax, (10.6 - 0.9, 5.45 - BOX_H / 2), (4.9, 2.0))
    draw_arrow(ax, (10.6 - 0.3, 5.45 - BOX_H / 2), (7.7, 2.0))
    draw_arrow(ax, (10.6, 5.45 - BOX_H / 2), (10.9, 2.0))

    ax.text(0.2, 6.5, "Two engines, one catalog, one snapshot",
            ha="left", va="center", fontsize=14, color=INK, fontfamily="Lora")
    ax.text(0.2, 0.75,
            "Scalar ops metrics and model-feature distributions need different checks; the catalog\n"
            "configures both, and nothing downstream ever recomputes a detector or reruns popmon.",
            ha="left", va="center", fontsize=10, color=GREY, fontfamily="Lato")

    fig.tight_layout()
    out_path = BASE / "assets" / "architecture.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
