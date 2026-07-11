"""Unit test for the top-words-per-topic extraction, a pure function
over an NMF components matrix."""
import numpy as np

from ticket_topics import top_words_per_topic


class _FakeNMF:
    def __init__(self, components):
        self.components_ = components


def test_top_words_ranks_by_weight_descending():
    # topic 0: "refund" has the highest weight, should come first
    components = np.array([
        [0.1, 0.9, 0.3],   # -> feature_names[1] first, then [2], then [0]
        [0.8, 0.05, 0.2],  # -> feature_names[0] first
    ])
    feature_names = ["refund", "late", "app"]
    words = top_words_per_topic(_FakeNMF(components), feature_names, n_top=2)
    assert words[0] == ["late", "app"]
    assert words[1] == ["refund", "app"]


def test_n_top_limits_returned_words():
    components = np.array([[0.5, 0.4, 0.3, 0.2, 0.1]])
    feature_names = ["a", "b", "c", "d", "e"]
    words = top_words_per_topic(_FakeNMF(components), feature_names, n_top=3)
    assert words[0] == ["a", "b", "c"]
