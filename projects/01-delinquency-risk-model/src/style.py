"""Shared plotting style for this project's charts (editorial/report look:
left-aligned bold titles, minimal chrome, consistent palette)."""
import matplotlib.pyplot as plt
import seaborn as sns

NAVY = "#1B2A4A"
TEAL = "#2E7D8C"
MINT = "#4FB0A5"
AMBER = "#E8A33D"
CORAL = "#D8574A"
GREY = "#8C93A6"
LIGHT_GREY = "#D8DCE5"

PALETTE = [NAVY, TEAL, MINT, AMBER, CORAL]
DIVERGING = [TEAL, CORAL]

FONT = "DejaVu Sans"


def set_style():
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "axes.edgecolor": LIGHT_GREY,
        "axes.linewidth": 1.0,
        "text.color": "#222222",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.8,
    })


def style_ax(ax, title=None, subtitle=None, xlabel=None, ylabel=None, grid_axis="y"):
    """Apply a consistent editorial look: bold left-aligned title, optional
    grey subtitle, only the requested gridlines, no top/right spines."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LIGHT_GREY)

    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)

    if title:
        y = 1.10 if subtitle else 1.04
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold",
                      color="#111111", y=y, pad=6)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=11,
                 color=GREY, ha="left", va="bottom")
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8)
    return ax


def savefig(fig, path, dpi=170):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
