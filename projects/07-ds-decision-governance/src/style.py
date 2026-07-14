"""Shared plotting style for this project's diagram. Same house style
used across the other projects in this repo: serif headline, sans body,
muted single-accent color use."""
import matplotlib.pyplot as plt

INK = "#ECECEA"
SLATE = "#8FBBDB"
MUTED_TEAL = "#82C2B7"
MUTED_AMBER = "#D8AD72"
MUTED_RED = "#C97B6E"
GREY = "#9C9C97"
LIGHT_GREY = "#34383E"
BG = "#1E2124"

SANS = "Lato"
SERIF = "Lora"


def set_style():
    plt.rcParams.update({
        "font.family": SANS,
        "text.color": INK,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    })
