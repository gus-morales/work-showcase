"""
Hyperparameter search for the GBM, using time-series cross-validation
instead of a single train/val split.

Why this matters: train.py's original hyperparameters were hand-picked
("sane defaults"), and the whole pipeline was validated on one chronological
split. A single split means the reported AUC could just be a lucky (or
unlucky) draw for that particular 3-month window. TimeSeriesSplit fits on an
expanding window and validates on the months immediately after it, several
times over, so the search picks hyperparameters that generalize across time
rather than to one snapshot.

Scope: search only uses months 1-18 (the "train" region). The calibration
window (19-21) and test window (22-24, which includes the macro shock) are
never touched here, so there's no leakage into the numbers reported in
train.py.

Run:
    python src/tune.py
Writes:
    reports/best_params.json
    reports/figures/cv_search_results.png
    reports/figures/cv_fold_stability.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from features import build_design_matrix, engineer_features
from style import set_style, style_ax, savefig, add_footnote, SLATE, MUTED_RED, GREY
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
set_style()

N_SPLITS = 5
N_ITER = 30
SEED = 42

PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6, 7, None],
    "learning_rate": uniform(0.02, 0.18),
    "max_iter": randint(100, 500),
    "l2_regularization": uniform(0.0, 2.0),
    "max_leaf_nodes": [15, 31, 63, 127, None],
    "min_samples_leaf": randint(10, 60),
}


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)
    train_df = df[df.origination_month <= 18].sort_values("origination_month")

    X_all, feature_names = build_design_matrix(df)
    X_train = X_all.loc[train_df.index]
    y_train = train_df["delinquent_30dpd"].values

    print(f"Tuning on {len(train_df):,} loans, months 1-18, sorted chronologically")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    base_model = HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced")

    search = RandomizedSearchCV(
        base_model, PARAM_DISTRIBUTIONS, n_iter=N_ITER, scoring="roc_auc",
        cv=tscv, random_state=SEED, n_jobs=-1, refit=False,
    )
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    best_row = results.iloc[0]
    best_params = {k.replace("param_", ""): v for k, v in best_row.items() if k.startswith("param_")}
    # clean up numpy types for JSON
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}

    print(f"Best CV AUC: {best_row['mean_test_score']:.4f} +/- {best_row['std_test_score']:.4f}")
    print("Best params:", json.dumps(best_params, indent=2))

    # --- Figure 1: top 10 candidate configs, mean CV AUC with std error bars ---
    top10 = results.head(10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(top10))[::-1]
    colors = [MUTED_RED if i == 0 else SLATE for i in range(len(top10))]
    ax.barh(y_pos, top10["mean_test_score"], xerr=top10["std_test_score"],
            color=colors, height=0.6, zorder=3, error_kw={"ecolor": GREY, "elinewidth": 1.2, "capsize": 3})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"config {i+1}" for i in range(len(top10))])
    ax.set_xlim(top10["mean_test_score"].min() - 0.03, top10["mean_test_score"].max() + 0.03)
    style_ax(ax, title="Top 10 candidate configurations",
             subtitle=f"Randomized search, {N_ITER} candidates x {N_SPLITS}-fold time-series CV",
             xlabel="Mean CV AUC (± 1 std across folds)", grid_axis="x")
    add_footnote(fig, "Source: synthetic BNPL loan data · CV on months 1-18 only (train window) · scoring = ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_search_results.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-fold AUC for the best config (temporal stability check) ---
    split_cols = [c for c in results.columns if c.startswith("split") and c.endswith("_test_score")]
    fold_scores = best_row[split_cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(fold_scores) + 1), fold_scores, marker="o", markersize=6, color=SLATE, linewidth=1.6)
    ax.axhline(fold_scores.mean(), ls="--", color=GREY, linewidth=1.2, label=f"Mean = {fold_scores.mean():.3f}")
    style_ax(ax, title="Best configuration's AUC across CV folds",
             subtitle="Each fold validates on the months right after an expanding training window",
             xlabel="CV fold (chronological order)", ylabel="Validation AUC", grid_axis="y")
    ax.set_xticks(range(1, len(fold_scores) + 1))
    ax.legend(loc="lower right")
    add_footnote(fig, "Source: synthetic BNPL loan data · CV on months 1-18 only (train window)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cv_fold_stability.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    out = {
        "best_params": best_params,
        "cv_mean_auc": round(float(best_row["mean_test_score"]), 4),
        "cv_std_auc": round(float(best_row["std_test_score"]), 4),
        "cv_fold_auc": [round(float(s), 4) for s in fold_scores],
        "n_candidates_searched": N_ITER,
        "n_cv_splits": N_SPLITS,
    }
    with open(BASE / "reports" / "best_params.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote reports/best_params.json, reports/figures/cv_search_results.png, "
          f"reports/figures/cv_fold_stability.png")


if __name__ == "__main__":
    main()
