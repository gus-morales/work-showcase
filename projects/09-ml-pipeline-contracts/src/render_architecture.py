"""Renders the Model Scope -> five-stage contract chain as a diagram
(not a chart, there's no data behind it) to assets/architecture.png,
reused as the portfolio thumbnail for this project.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from style import set_style, BG, GREY, INK, LIGHT_GREY, MUTED_AMBER, MUTED_RED, MUTED_TEAL

BASE = Path(__file__).resolve().parents[1]
set_style()


def draw_box(ax, x, y, label, color, w=2.5, h=0.85, fontsize=10.5):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        linewidth=1.2, edgecolor=color, facecolor=LIGHT_GREY,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=INK,
             fontfamily="Lato", linespacing=1.4)
    return w, h


def draw_arrow(ax, xy_from, xy_to, color=GREY, label=None, label_offset=(0, 0), fontsize=8.7):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3, shrinkA=6, shrinkB=6),
    )
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label, ha="center", va="center",
                fontsize=fontsize, color=color, fontfamily="Lato")


def main():
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0.2, 6.8)
    ax.axis("off")

    scope_w, _ = draw_box(ax, 2.0, 6.0, "Model Scope Document\nstatus: frozen", MUTED_AMBER, w=3.3)
    bindings_w, _ = draw_box(ax, 2.0, 4.6, "bindings.yaml\nfeature_spec.yaml", GREY, w=3.3, h=0.7, fontsize=9.5)

    stage_names = ["① EDA", "② Data", "③ Features", "④ Training", "⑤ Validation"]
    xs = [4.9, 6.85, 8.8, 10.75, 12.9]
    for x, name in zip(xs, stage_names):
        draw_box(ax, x, 6.0, name, MUTED_TEAL, w=1.7)

    draw_arrow(ax, (2.0 + scope_w / 2 - 0.2, 6.0), (4.9 - 0.9, 6.0))
    for x_from, x_to in zip(xs, xs[1:]):
        draw_arrow(ax, (x_from + 0.85, 6.0), (x_to - 0.85, 6.0))

    for x in xs[1:]:
        draw_box(ax, x - 1.0, 4.6, "verify_*()", MUTED_RED, w=1.4, h=0.55, fontsize=8.5)
        draw_arrow(ax, (x - 1.0, 6.0 - 0.45), (x - 1.0, 4.6 + 0.3), color=MUTED_RED)

    draw_box(ax, 7.25, 2.2,
             "a broken handoff stops the chain here,\nnamed by stage and field, before Training ever runs",
             MUTED_RED, w=6.6, h=1.0, fontsize=9.5)
    draw_arrow(ax, (7.25, 2.2 + 0.55), (7.25, 4.6 - 0.35), color=MUTED_RED)

    fig.tight_layout()
    assets_dir = BASE / "assets"
    assets_dir.mkdir(exist_ok=True)
    fig.savefig(assets_dir / "architecture.png", dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {assets_dir / 'architecture.png'}")


if __name__ == "__main__":
    main()
