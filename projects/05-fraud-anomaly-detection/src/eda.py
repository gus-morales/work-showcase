"""Exploratory analysis on the raw transaction data, before any model gets
trained: saves charts to reports/figures and a numeric summary to
reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    tx = pd.read_csv(BASE / "data" / "transactions.csv")
    SOURCE = f"Source: synthetic BNPL transaction data (src/generate_data.py) · n = {len(tx):,} transactions"

    n_genuine = int((tx["is_fraud"] == 0).sum())
    n_fraud = int((tx["is_fraud"] == 1).sum())
    fraud_rate = tx["is_fraud"].mean()

    # 1. Class imbalance, on a linear scale on purpose: the fraud bar is
    # supposed to look almost invisible next to the genuine one, since
    # that's the actual shape of the problem every metric choice and
    # threshold decision later has to work around.
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ["Genuine", "Fraud"]
    vals = [n_genuine, n_fraud]
    ax.bar(bars, vals, color=[SLATE, MUTED_RED], width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + len(tx) * 0.01, f"{v:,} ({v / len(tx):.1%})", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title=f"Fraud is {fraud_rate:.1%} of transactions, genuine dwarfs it on a linear scale",
             subtitle="Transaction count by class",
             ylabel="Transactions")
    savefig(fig, FIG_DIR / "class_imbalance.png", footnote=SOURCE)

    # 2. Fraud rate by device recognition
    rates = tx.groupby("is_new_device")["is_fraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = ["Recognized device", "New/unrecognized device"]
    ax.bar(labels, rates.values, color=[SLATE, MUTED_RED], width=0.5, zorder=3)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="An unrecognized device is one of the strongest single fraud signals",
             subtitle="Fraud rate by device recognition", ylabel="Fraud rate (%)")
    savefig(fig, FIG_DIR / "fraud_rate_by_device.png", footnote=SOURCE)

    # 3. Fraud rate by recent transaction velocity
    tx["velocity_bucket"] = pd.cut(tx["transactions_last_1h"], [-1, 0, 1, 10],
                                    labels=["0", "1", "2+"])
    v_rates = tx.groupby("velocity_bucket", observed=True)["is_fraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(v_rates.index.astype(str), v_rates.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(v_rates.values):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Fraud rate climbs with recent transaction velocity",
             subtitle="Fraud rate by transactions in the 1-hour window before this one",
             xlabel="Transactions in the last hour", ylabel="Fraud rate (%)")
    savefig(fig, FIG_DIR / "fraud_rate_by_velocity.png", footnote=SOURCE)

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Transactions: {len(tx):,}, Customers: {tx['customer_id'].nunique():,}\n")
    lines.append(f"- Fraud rate: {fraud_rate:.3%} ({n_fraud:,} fraud vs. {n_genuine:,} genuine, "
                 f"a {n_genuine / n_fraud:.0f}:1 ratio)\n")
    lines.append(f"- Fraud rate, new/unrecognized device vs. recognized: "
                 f"{rates.loc[1]:.2f}% vs. {rates.loc[0]:.2f}%\n")
    lines.append(f"- Fraud rate, 2+ transactions in the last hour vs. 0: "
                 f"{v_rates.loc['2+']:.2f}% vs. {v_rates.loc['0']:.2f}%\n")
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
