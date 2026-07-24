"""Unit tests for the segment-labeling logic: a pure function over a
small hand-built cluster-mean profile, independent of GaussianMixture
itself. Mirrors project 03's test_segmentation.py, which tests the same
kind of rank-based labeling for its KMeans/RFM segments."""
import pandas as pd

from segmentation import label_segments, NAME_SETS


def test_best_and_worst_segment_labeled_correctly_for_three_clusters():
    # cluster 0 = best on every dimension (recent, frequent, high spend)
    # cluster 2 = worst on every dimension
    profile = pd.DataFrame({
        "recency_days": [10, 80, 170],
        "frequency_90d": [12, 5, 1],
        "monetary_90d": [900, 300, 20],
    }, index=[0, 1, 2])
    labels = label_segments(profile)
    assert labels[0] == "High-Value Active"
    assert labels[2] == "Lapsed/Dormant"
    assert labels[1] == "Steady Regular"


def test_label_count_matches_cluster_count():
    for k in [2, 3, 4, 5, 6]:
        profile = pd.DataFrame({
            "recency_days": list(range(10, 10 + k * 30, 30)),
            "frequency_90d": list(range(k * 3, 0, -3)),
            "monetary_90d": list(range(k * 500, 0, -500)),
        })
        labels = label_segments(profile)
        assert set(labels.values()) == set(NAME_SETS[k])
        assert len(labels) == k


def test_falls_back_gracefully_for_unmapped_cluster_count():
    profile = pd.DataFrame({
        "recency_days": [10, 40, 70, 100, 130, 160, 179],
        "frequency_90d": [15, 12, 9, 6, 4, 2, 1],
        "monetary_90d": [1200, 900, 700, 500, 300, 150, 20],
    })
    labels = label_segments(profile)  # k=7, not in NAME_SETS
    assert len(labels) == 7
    assert all(v.startswith("Segment") for v in labels.values())
