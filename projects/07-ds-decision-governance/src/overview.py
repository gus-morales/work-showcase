"""What's in the decision log: composition by artifact type, impact
level, and final status. Context for the governance analysis that
follows, not a separate analytical phase, saves charts to
reports/figures and a numeric summary to reports/overview_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()

IMPACT_ORDER = ["low", "medium", "high"]
STATUS_ORDER = ["closed", "reverted", "abandoned"]


def main():
    df = pd.read_csv(BASE / "data" / "decision_log.csv")
    SOURCE = f"Source: synthetic decision log (src/generate_data.py) · n = {len(df):,} decisions"

    # 1. Volume by artifact type
    type_counts = df["artifact_type"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(type_counts.index.str.replace("_", " "), type_counts.values, color=SLATE, width=0.55, zorder=3)
    for i, v in enumerate(type_counts.values):
        ax.text(i, v + len(df) * 0.008, f"{v:,}", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Dashboard and pipeline changes make up half the log",
             subtitle="Decision count by artifact type",
             ylabel="Decisions")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, FIG_DIR / "volume_by_artifact_type.png", footnote=SOURCE)

    # 2. Impact level mix
    impact_counts = df["impact_level"].value_counts().reindex(IMPACT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(impact_counts.index, impact_counts.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(impact_counts.values):
        ax.text(i, v + len(df) * 0.008, f"{v:,} ({v / len(df):.0%})", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Most decisions are low impact",
             subtitle="Decision count by impact level",
             xlabel="Impact level", ylabel="Decisions")
    savefig(fig, FIG_DIR / "impact_level_mix.png", footnote=SOURCE)

    # 3. Status mix
    status_counts = df["status"].value_counts().reindex(STATUS_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(status_counts.index, status_counts.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(status_counts.values):
        ax.text(i, v + len(df) * 0.008, f"{v:,} ({v / len(df):.0%})", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Most decisions close out without a rollback",
             subtitle="Decision count by final status",
             xlabel="Status", ylabel="Decisions")
    savefig(fig, FIG_DIR / "status_mix.png", footnote=SOURCE)

    # Summary markdown
    lines = ["# Decision log overview\n"]
    lines.append(f"- Decisions: {len(df):,}\n")
    lines.append(f"- Artifact type counts:\n{type_counts.to_string()}\n")
    lines.append(f"- Impact level mix:\n{(impact_counts / len(df)).round(3).to_string()}\n")
    lines.append(f"- Status mix:\n{(status_counts / len(df)).round(3).to_string()}\n")
    (BASE / "reports" / "overview_summary.md").write_text("\n".join(lines))
    print("Overview complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
