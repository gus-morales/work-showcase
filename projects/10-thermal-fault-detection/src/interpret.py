"""Interpretability for both models: Grad-CAM for the CNN (does it actually look at
the fault, or something else?) and feature importance for the classical model."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from style import BG, GREY, INK, MUTED_AMBER, MUTED_RED, MUTED_TEAL, savefig, set_style, style_ax  # noqa: E402
from train_cnn import normalize  # noqa: E402

tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "thermal_images.npz"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODEL_PATH = REPORTS_DIR / "model_cnn.keras"
SPLIT_PATH = REPORTS_DIR / "split_indices.npz"
METRICS_CLASSICAL_PATH = REPORTS_DIR / "metrics_classical.json"
GRADCAM_REPORT_PATH = REPORTS_DIR / "gradcam_alignment.json"

THERMAL_CMAP = LinearSegmentedColormap.from_list("thermal", [BG, MUTED_RED, MUTED_AMBER])
LAST_CONV_LAYER = "conv3"


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["images"], d["labels"], d["fault_x"], d["fault_y"]


def grad_cam(model, image, layer_name=LAST_CONV_LAYER):
    """Standard Grad-CAM: weight the last conv layer's feature maps by how much each
    channel pushes the fault-probability output up, then collapse to one heatmap."""
    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output]
    )
    x = tf.convert_to_tensor(image[np.newaxis, ...])
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(x)
        loss = prediction[:, 0]
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., np.newaxis], image.shape[:2])[..., 0]
    return cam.numpy(), float(prediction[0, 0])


def heatmap_peak_location(cam):
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return float(x), float(y)


def plot_gradcam_examples(model, images, labels, fault_x, fault_y, test_idx):
    set_style()
    fault_test_idx = [i for i in test_idx if labels[i] == 1]
    rng = np.random.default_rng(0)
    chosen = rng.choice(fault_test_idx, size=4, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    normed = normalize(images)

    for col, idx in enumerate(chosen):
        img = normed[idx][..., np.newaxis]
        cam, pred = grad_cam(model, img)

        ax = axes[0, col]
        ax.imshow(images[idx], cmap=THERMAL_CMAP, vmin=15, vmax=70)
        ax.scatter([fault_x[idx]], [fault_y[idx]], marker="x", s=70, color=MUTED_TEAL, linewidths=2)
        ax.set_title(f"p(fault) = {pred:.2f}", fontsize=9.5, color=GREY, fontfamily="Lato")
        ax.axis("off")

        ax2 = axes[1, col]
        ax2.imshow(images[idx], cmap=THERMAL_CMAP, vmin=15, vmax=70)
        ax2.imshow(cam, cmap="inferno", alpha=0.55)
        ax2.axis("off")

    axes[0, 0].text(-0.35, 0.5, "Thermal image\n(x = true fault)", transform=axes[0, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=10.5, color=INK, fontfamily="Lora")
    axes[1, 0].text(-0.35, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=10.5, color=INK, fontfamily="Lora")

    fig.suptitle("Grad-CAM lights up the same spot a human would circle",
                  fontsize=14.5, color=INK, fontfamily="Lora", x=0.02, ha="left", y=1.02)
    savefig(fig, FIG_DIR / "gradcam_examples.png",
            footnote="Four held-out fault images with the model's predicted probability and Grad-CAM overlay.")


def plot_classical_feature_importance():
    set_style()
    metrics = json.loads(METRICS_CLASSICAL_PATH.read_text())
    importances = metrics["feature_importances"]
    items = sorted(importances.items(), key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(names, values, color=MUTED_TEAL, zorder=3)
    style_ax(
        ax,
        title="The classical model leans almost entirely on peak temperature",
        subtitle="Feature importance, OpenCV-features + gradient-boosted classifier",
        xlabel="Importance",
        grid_axis="x",
    )
    savefig(fig, FIG_DIR / "classical_feature_importance.png",
            footnote="Gradient-boosted classifier feature importances, held-out test set.")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, fault_x, fault_y = load_data()
    split = np.load(SPLIT_PATH)
    test_idx = split["test_idx"]

    model = tf.keras.models.load_model(MODEL_PATH)

    plot_gradcam_examples(model, images, labels, fault_x, fault_y, test_idx)
    plot_classical_feature_importance()

    # quantitative check: does Grad-CAM's peak actually land near the true fault?
    normed = normalize(images)
    fault_test_idx = [i for i in test_idx if labels[i] == 1]
    distances = []
    for idx in fault_test_idx:
        img = normed[idx][..., np.newaxis]
        cam, _ = grad_cam(model, img)
        px, py = heatmap_peak_location(cam)
        dist = float(np.hypot(px - fault_x[idx], py - fault_y[idx]))
        distances.append(dist)
    distances = np.array(distances)

    # a random guess anywhere in a 64x64 image has this expected distance to the fault
    rng = np.random.default_rng(0)
    random_px = rng.uniform(0, 64, size=(2000, 2))
    random_dist_baseline = float(np.mean([
        np.hypot(random_px[:, 0] - fx, random_px[:, 1] - fy).mean()
        for fx, fy in zip(fault_x[fault_test_idx], fault_y[fault_test_idx])
    ]))

    report = {
        "n_fault_test_images": len(fault_test_idx),
        "mean_gradcam_peak_distance_px": float(distances.mean()),
        "median_gradcam_peak_distance_px": float(np.median(distances)),
        "random_guess_expected_distance_px": random_dist_baseline,
        "image_size_px": 64,
    }
    GRADCAM_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Grad-CAM peak vs. true fault location, {len(fault_test_idx)} held-out fault images:")
    print(f"  mean distance:   {report['mean_gradcam_peak_distance_px']:.1f} px")
    print(f"  median distance: {report['median_gradcam_peak_distance_px']:.1f} px")
    print(f"  random-guess baseline: {report['random_guess_expected_distance_px']:.1f} px")


if __name__ == "__main__":
    main()
