"""Shared plotting style for this project's charts. Restrained, quiet
report style: serif headline, sans body, muted single-accent color use,
minimal on-chart annotation."""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

INK = "#2B2B2B"
SLATE = "#3D5A73"
MUTED_TEAL = "#5E8C86"
MUTED_AMBER = "#B98A3D"
MUTED_RED = "#A8493D"
GREY = "#8A8A8A"
LIGHT_GREY = "#E3E3E1"

PALETTE = [SLATE, MUTED_TEAL, MUTED_AMBER, MUTED_RED, GREY]

SANS = "Lato"
SERIF = "Lora"


def set_style():
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": 11.5,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#4a4a4a",
        "axes.edgecolor": LIGHT_GREY,
        "axes.linewidth": 0.9,
        "text.color": INK,
        "xtick.color": "#6a6a6a",
        "ytick.color": "#6a6a6a",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.7,
    })


def style_ax(ax, title=None, subtitle=None, xlabel=None, ylabel=None, grid_axis="y"):
    """Quiet report look: serif title (not oversized/bold-black), grey
    subtitle, minimal gridlines, no top/right spines."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(LIGHT_GREY)

    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0, linewidth=0.7)
        ax.set_axisbelow(True)

    if title:
        y = 1.13 if subtitle else 1.05
        ax.set_title(title, loc="left", fontsize=14.5, fontweight="normal",
                      fontfamily=SERIF, color=INK, y=y, pad=4)
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=10.5,
                 color=GREY, ha="left", va="bottom", fontfamily=SANS)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8)
    return ax


def add_footnote(fig, text):
    """Small grey source/sample-size line pinned to the bottom-left of the
    figure, in the style of FT/Economist chart footers."""
    fig.text(0.01, -0.04, text, fontsize=8.5, color=GREY, ha="left", va="top", fontfamily=SANS)


def savefig(fig, path, dpi=170, footnote=None):
    if footnote:
        add_footnote(fig, footnote)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
