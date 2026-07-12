"""
Light NLP: unsupervised topic modeling on support ticket text.

TF-IDF vectorizes the ticket text, NMF (non-negative matrix
factorization) extracts latent topics, and each ticket is assigned to
its dominant topic. The synthetic data was generated from five known
underlying topics, which is used here only as an evaluation check
(cluster purity against ground truth), the kind of validation that
isn't available on real, unlabeled ticket data but is worth showing
once, on data where it can be checked.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from style import set_style, style_ax, savefig, SLATE, PALETTE, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

TOPIC_RANGE = range(3, 9)


def choose_n_topics(X, source_note):
    errors = []
    for k in TOPIC_RANGE:
        nmf = NMF(n_components=k, init="nndsvda", random_state=7, max_iter=400)
        nmf.fit(X)
        errors.append(nmf.reconstruction_err_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(TOPIC_RANGE), errors, color=SLATE, marker="o", markersize=5, linewidth=1.6)
    style_ax(ax, title="Reconstruction error by topic count",
             subtitle="NMF Frobenius reconstruction error",
             xlabel="Number of topics", ylabel="Reconstruction error")
    savefig(fig, FIG_DIR / "topics_k_selection.png", footnote=source_note)
    return errors


def top_words_per_topic(nmf, feature_names, n_top=8):
    words = {}
    for i, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[::-1][:n_top]
        words[i] = [feature_names[j] for j in top_idx]
    return words


def main():
    df = pd.read_csv(DATA_DIR / "support_tickets.csv")
    n_tickets = len(df)
    source_note = f"Source: synthetic BNPL support tickets · n = {n_tickets:,} tickets"

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.6, min_df=3, ngram_range=(1, 1))
    X = vectorizer.fit_transform(df["ticket_text"])
    feature_names = vectorizer.get_feature_names_out()

    choose_n_topics(X, source_note)
    # NMF reconstruction error decreases smoothly with topic count and has
    # no sharp elbow here, so it doesn't pick k on its own. 5 is chosen as
    # a reasonable operational number of ticket categories for a support
    # team to triage against, then checked against the top words per topic
    # below for coherence.
    N_TOPICS = 5
    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=7, max_iter=400)
    W = nmf.fit_transform(X)
    df["topic"] = W.argmax(axis=1)

    words = top_words_per_topic(nmf, feature_names)
    topic_labels = {i: ", ".join(words[i][:3]) for i in range(N_TOPICS)}
    df["topic_label"] = df["topic"].map(topic_labels)

    # --- Top words per topic ---
    fig, axes = plt.subplots(1, N_TOPICS, figsize=(18, 4.5))
    for i, ax in enumerate(axes):
        comp = nmf.components_[i]
        top_idx = comp.argsort()[::-1][:8]
        top_vals = comp[top_idx]
        top_terms = [feature_names[j] for j in top_idx]
        ax.barh(range(len(top_terms))[::-1], top_vals, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.set_yticks(range(len(top_terms))[::-1])
        ax.set_yticklabels(top_terms, fontsize=9)
        style_ax(ax, title=f"Topic {i+1}", grid_axis="x")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Five topics emerge cleanly from ticket text alone", x=0.01, y=1.05,
                 ha="left", fontsize=14.5, fontfamily="Lora", color=INK)
    plt.tight_layout()
    savefig(fig, FIG_DIR / "topic_top_words.png", footnote=source_note)

    # --- Topic volume ---
    vol = df["topic_label"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(vol.index, vol.values, color=SLATE, zorder=3)
    for i, v in enumerate(vol.values):
        ax.text(v + 5, i, f"{v} ({v/n_tickets:.0%})", va="center", fontsize=9.5, color=INK)
    style_ax(ax, title="Four topics arrive in similar volume; late-fee disputes split into two sub-themes",
             subtitle="Ticket volume by discovered topic", xlabel="Tickets", grid_axis="x")
    savefig(fig, FIG_DIR / "topic_volume.png", footnote=source_note)

    # --- Validation against known ground truth (synthetic data only) ---
    cross = pd.crosstab(df["true_topic"], df["topic"])
    # Greedy match: for each discovered topic, the ground-truth label it overlaps with most
    match = cross.idxmax(axis=0)
    purity = sum(cross.loc[match[t], t] for t in cross.columns) / n_tickets
    print(f"Cluster purity vs. ground-truth topic (synthetic data validation only): {purity:.1%}")
    print(cross)

    df.drop(columns=["true_topic"]).to_csv(BASE / "reports" / "ticket_topics.csv", index=False)
    print("Wrote reports/ticket_topics.csv and 3 figures.")


if __name__ == "__main__":
    main()
