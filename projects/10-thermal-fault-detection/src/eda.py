"""Exploratory look at the synthetic thermal-image dataset: class balance, example
images, and how much a naive brightest-pixel rule would confuse benign warm spots
with real faults."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from style import BG, INK, GREY, MUTED_AMBER, MUTED_RED, MUTED_TEAL, savefig, set_style, style_ax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "thermal_images.npz"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_summary.md"

THERMAL_CMAP = LinearSegmentedColormap.from_list("thermal", [BG, MUTED_RED, MUTED_AMBER])


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["images"], d["labels"], d["component"], d["severity"]


def plot_class_imbalance(labels):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    ax.bar(["Healthy", "Fault"], counts, color=[MUTED_TEAL, MUTED_RED], width=0.55, zorder=3)
    for i, c in enumerate(counts):
        ax.text(i, c + 15, f"{c:,}", ha="center", fontsize=11, color=INK)
    style_ax(
        ax,
        title="Most inspection images show a healthy component",
        subtitle=f"{counts[1] / sum(counts):.1%} of {sum(counts):,} thermal images carry a fault",
        ylabel="Images",
    )
    savefig(fig, FIG_DIR / "class_imbalance.png", footnote="Synthetic thermal-inspection dataset.")


def plot_example_images(images, labels, component):
    set_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    vmin, vmax = 15, 70

    healthy_idx = rng.choice(np.where(labels == 0)[0], size=4, replace=False)
    fault_idx = rng.choice(np.where(labels == 1)[0], size=4, replace=False)

    for col, idx in enumerate(healthy_idx):
        ax = axes[0, col]
        ax.imshow(images[idx], cmap=THERMAL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(str(component[idx]).replace("_", " "), fontsize=9.5, color=GREY, fontfamily="Lato")
        ax.axis("off")
    for col, idx in enumerate(fault_idx):
        ax = axes[1, col]
        ax.imshow(images[idx], cmap=THERMAL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(str(component[idx]).replace("_", " "), fontsize=9.5, color=GREY, fontfamily="Lato")
        ax.axis("off")

    axes[0, 0].text(-0.35, 0.5, "Healthy", transform=axes[0, 0].transAxes, rotation=90,
                     va="center", ha="center", fontsize=12, color=INK, fontfamily="Lora")
    axes[1, 0].text(-0.35, 0.5, "Fault", transform=axes[1, 0].transAxes, rotation=90,
                     va="center", ha="center", fontsize=12, color=INK, fontfamily="Lora")

    fig.suptitle("A fault image looks like a healthy one plus one extra bright spot",
                  fontsize=14.5, color=INK, fontfamily="Lora", x=0.02, ha="left", y=1.02)
    savefig(fig, FIG_DIR / "example_images.png",
            footnote="Same fixed color scale (15-70) across all eight images.")


def plot_intensity_overlap(images, labels):
    set_style()
    max_intensity = images.reshape(len(images), -1).max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(max_intensity.min(), max_intensity.max(), 40)
    ax.hist(max_intensity[labels == 0], bins=bins, color=MUTED_TEAL, alpha=0.75,
            label="Healthy", zorder=3)
    ax.hist(max_intensity[labels == 1], bins=bins, color=MUTED_RED, alpha=0.75,
            label="Fault", zorder=3)
    style_ax(
        ax,
        title="Brightness alone doesn't cleanly separate healthy from fault",
        subtitle="A mild fault's peak temperature overlaps healthy images with a benign warm spot",
        xlabel="Peak temperature in the image",
        ylabel="Images",
    )
    ax.legend()
    savefig(fig, FIG_DIR / "intensity_overlap.png",
            footnote="Peak-pixel temperature, healthy vs. fault, full dataset.")


def naive_threshold_stats(images, labels):
    """How well would 'flag the brightest images' do on its own?"""
    max_intensity = images.reshape(len(images), -1).max(axis=1)
    threshold = np.percentile(max_intensity[labels == 1], 25)  # catch the top 75% of faults
    predicted = max_intensity > threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "false_positives": fp, "true_positives": tp}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, component, severity = load_data()

    plot_class_imbalance(labels)
    plot_example_images(images, labels, component)
    plot_intensity_overlap(images, labels)
    stats = naive_threshold_stats(images, labels)

    lines = [
        "# EDA summary: thermal fault detection",
        "",
        f"- Total images: {len(labels):,}",
        f"- Fault rate: {labels.mean():.1%}",
        "",
        "## Naive brightest-pixel threshold",
        f"- Threshold set to catch 75% of faults: peak temperature > {stats['threshold']:.1f}",
        f"- Resulting precision: {stats['precision']:.1%}",
        f"- Resulting recall: {stats['recall']:.1%}",
        f"- False positives: {stats['false_positives']} healthy images incorrectly flagged",
        "",
        "A single brightness threshold either misses mild faults or flags a large share "
        "of healthy images with a benign warm spot. The rest of this project compares two "
        "learned approaches against this baseline.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
