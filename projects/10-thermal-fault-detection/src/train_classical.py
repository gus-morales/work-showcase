"""Classical computer-vision baseline: OpenCV hotspot features + component type,
fed into a gradient-boosted classifier. No raw pixels, no deep learning -- this is the
"a domain expert's rule of thumb, formalized" path, and the baseline the CNN in
train_cnn.py has to beat."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from features import extract_features_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "thermal_images.npz"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = REPORTS_DIR / "model_classical.pkl"
METRICS_PATH = REPORTS_DIR / "metrics_classical.json"
SPLIT_PATH = REPORTS_DIR / "split_indices.npz"

SEED = 10
TEST_SIZE = 0.25


def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["images"], d["labels"], d["component"], d["severity"]


def make_split(labels, seed=SEED, test_size=TEST_SIZE):
    """Stratified train/test split, shared with train_cnn.py so both models are
    scored on the exact same held-out images."""
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    return train_idx, test_idx


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    images, labels, component, severity = load_data()

    if SPLIT_PATH.exists():
        split = np.load(SPLIT_PATH)
        train_idx, test_idx = split["train_idx"], split["test_idx"]
    else:
        train_idx, test_idx = make_split(labels)
        np.savez(SPLIT_PATH, train_idx=train_idx, test_idx=test_idx)

    features = extract_features_batch(images, component)
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = GradientBoostingClassifier(random_state=SEED)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    base_rate = float(y_test.mean())

    importances = dict(zip(features.columns, model.feature_importances_.round(4).tolist()))

    metrics = {
        "model": "classical (OpenCV features + GBM)",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_fault_rate": base_rate,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "feature_importances": importances,
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Test set: {len(test_idx)} images, {base_rate:.1%} fault rate")
    print(f"PR-AUC:  {pr_auc:.3f}  ({pr_auc / base_rate:.1f}x base rate)")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("Feature importances:")
    for name, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {imp}")


if __name__ == "__main__":
    main()
