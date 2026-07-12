"""Exploratory analysis on the raw customer/order data, before any cohort,
contribution, or CLV modeling: saves charts to reports/figures and a numeric
summary to reports/eda_summary.md."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, PALETTE, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()

CHANNEL_LABELS = {
    "partner_store": "Partner store", "referral": "Referral",
    "organic": "Organic", "paid_social": "Paid social",
}


def main():
    customers = pd.read_csv(BASE / "data" / "customers.csv")
    orders = pd.read_csv(BASE / "data" / "orders.csv")
    SOURCE = f"Source: synthetic BNPL order data (src/generate_data.py) · n = {len(customers):,} customers"

    orders_per_customer = orders.groupby("customer_id").size()

    # 1. Orders per customer (heavily right-skewed, the reason a
    # heterogeneous-purchase-rate model like BG/NBD fits better than a
    # single average-frequency assumption)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(1, orders_per_customer.clip(upper=30).max() + 2) - 0.5
    ax.hist(orders_per_customer.clip(upper=30), bins=bins, color=SLATE, zorder=3)
    style_ax(ax, title="Most customers order a handful of times; a long tail orders constantly",
             subtitle="Orders per customer (30+ grouped into the last bar)",
             xlabel="Orders per customer", ylabel="Customers")
    savefig(fig, FIG_DIR / "orders_per_customer_dist.png", footnote=SOURCE)

    # 2. Order value distribution (right-skewed, the reason Gamma-Gamma
    # models average order value instead of assuming it's constant)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(orders["order_value_usd"].clip(upper=2000), bins=40, color=SLATE, zorder=3)
    style_ax(ax, title="Order value varies far more than a single average implies",
             subtitle="Order value in USD (values above $2,000 grouped into the last bar)",
             xlabel="Order value (USD)", ylabel="Orders")
    savefig(fig, FIG_DIR / "order_value_dist.png", footnote=SOURCE)

    # 3. Overall acquisition channel mix (before looking at how that mix
    # shifts over time, or how much each channel is actually worth)
    mix = customers["acquisition_channel"].value_counts(normalize=True).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [CHANNEL_LABELS[c] for c in mix.index]
    ax.bar(labels, mix.values * 100, color=PALETTE[:len(mix)], width=0.55, zorder=3)
    for i, v in enumerate(mix.values):
        ax.text(i, v * 100 + 0.6, f"{v:.1%}", ha="center", fontsize=10, color=INK)
    style_ax(ax, title="Paid social is already the single largest acquisition channel",
             subtitle="Share of all customers acquired, by channel",
             ylabel="Share of customers (%)")
    savefig(fig, FIG_DIR / "channel_mix_overall.png", footnote=SOURCE)

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Customers: {len(customers):,}, Orders: {len(orders):,}\n")
    lines.append(f"- Orders per customer: mean {orders_per_customer.mean():.2f}, "
                 f"median {orders_per_customer.median():.0f}, "
                 f"90th percentile {orders_per_customer.quantile(0.9):.0f}, "
                 f"{(orders_per_customer == 1).mean():.1%} single-order customers\n")
    lines.append(f"- Order value (USD): mean {orders['order_value_usd'].mean():.2f}, "
                 f"median {orders['order_value_usd'].median():.2f}, "
                 f"90th percentile {orders['order_value_usd'].quantile(0.9):.2f}\n")
    lines.append(f"- Missing values: {int(customers.isna().sum().sum() + orders.isna().sum().sum())} "
                 f"(data contracts in src/contracts.py enforce this at generation time)\n")
    lines.append("\n## Acquisition channel mix\n")
    lines.append(mix.mul(100).round(1).astype(str).add(" %").to_markdown())
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
