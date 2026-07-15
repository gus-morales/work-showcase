"""Deep-learning path: a small CNN trained directly on raw pixels, no handcrafted
features and no component label. This is the head-to-head comparison against
train_classical.py's OpenCV-features-plus-GBM baseline, scored on the exact same
held-out images."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "thermal_images.npz"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = REPORTS_DIR / "model_cnn.keras"
METRICS_PATH = REPORTS_DIR / "metrics_cnn.json"
SPLIT_PATH = REPORTS_DIR / "split_indices.npz"
HISTORY_PATH = REPORTS_DIR / "cnn_training_history.json"

SEED = 10
VMIN, VMAX = 15.0, 70.0
VAL_SIZE = 0.15
EPOCHS = 20
BATCH_SIZE = 32


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["images"], d["labels"]


def normalize(images):
    """Same fixed vmin/vmax scale used in features.py, so both pipelines see the
    same physical range, just at float [0, 1] resolution instead of uint8."""
    clipped = np.clip(images, VMIN, VMAX)
    return ((clipped - VMIN) / (VMAX - VMIN)).astype("float32")


def build_model(input_shape=(64, 64, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same", name="conv1")(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="conv3")(x)
    features = x  # last conv feature map, used by Grad-CAM in interpret.py
    # max, not average: a fault is one small bright region against a much larger
    # normal background, and average-pooling dilutes exactly that kind of sparse
    # signal away. Max-pooling asks "is there at least one strongly-activated
    # location", which matches how these images actually differ.
    x = tf.keras.layers.GlobalMaxPooling2D()(features)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fault_prob")(x)
    return tf.keras.Model(inputs, outputs, name="thermal_fault_cnn")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    images, labels = load_data()
    images = normalize(images)[..., np.newaxis]  # (N, 64, 64, 1)

    split = np.load(SPLIT_PATH)
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    X_train_full, y_train_full = images[train_idx], labels[train_idx]
    X_test, y_test = images[test_idx], labels[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )

    # faults are ~12% of the data; weight them up so the loss doesn't just learn
    # to predict "healthy" every time
    fault_rate = float(y_train.mean())
    class_weight = {0: 1.0, 1: (1 - fault_rate) / fault_rate}

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    proba = model.predict(X_test, verbose=0).ravel()
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    base_rate = float(y_test.mean())

    metrics = {
        "model": "CNN (raw pixels, TensorFlow/Keras)",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_fault_rate": base_rate,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "epochs_run": len(history.history["loss"]),
    }

    model.save(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    HISTORY_PATH.write_text(json.dumps(
        {k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2
    ))

    print(f"Test set: {len(test_idx)} images, {base_rate:.1%} fault rate")
    print(f"Epochs run (early stopping): {metrics['epochs_run']}")
    print(f"PR-AUC:  {pr_auc:.3f}  ({pr_auc / base_rate:.1f}x base rate)")
    print(f"ROC-AUC: {roc_auc:.3f}")


if __name__ == "__main__":
    main()
