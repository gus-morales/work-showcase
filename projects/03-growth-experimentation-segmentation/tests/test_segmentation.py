"""Unit tests for the RFM cluster-labeling logic: a pure function over
a small hand-built cluster-mean profile, independent of KMeans itself."""
import pandas as pd

from segmentation import label_segments, NAME_SETS


def test_best_and_worst_segment_labeled_correctly_for_three_clusters():
    # cluster 0 = best on every dimension, cluster 2 = worst on every dimension
    profile = pd.DataFrame({
        "recency_days": [10, 100, 300],
        "frequency": [25, 10, 2],
        "monetary_usd": [12000, 4000, 800],
    }, index=[0, 1, 2])
    labels = label_segments(profile)
    assert labels[0] == "Champions"
    assert labels[2] == "Dormant"
    assert labels[1] == "Loyal"


def test_label_count_matches_cluster_count():
    for k in [2, 3, 4, 5, 6]:
        profile = pd.DataFrame({
            "recency_days": list(range(10, 10 + k * 50, 50)),
            "frequency": list(range(k * 5, 0, -5)),
            "monetary_usd": list(range(k * 1000, 0, -1000)),
        })
        labels = label_segments(profile)
        assert set(labels.values()) == set(NAME_SETS[k])
        assert len(labels) == k


def test_falls_back_gracefully_for_unmapped_cluster_count():
    profile = pd.DataFrame({
        "recency_days": [10, 50, 100, 150, 200, 250, 300],
        "frequency": [30, 25, 20, 15, 10, 5, 1],
        "monetary_usd": [9000, 7000, 5000, 4000, 3000, 2000, 500],
    })
    labels = label_segments(profile)  # k=7, not in NAME_SETS
    assert len(labels) == 7
    assert all(v.startswith("Segment") for v in labels.values())
