"""Unit tests for the power-analysis effect-size helper."""
import numpy as np
import pytest

from experiment_design import cohens_h


def test_zero_effect_size_when_proportions_equal():
    assert cohens_h(0.34, 0.34) == pytest.approx(0.0)


def test_antisymmetric_in_its_arguments():
    assert cohens_h(0.30, 0.40) == pytest.approx(-cohens_h(0.40, 0.30))


def test_positive_when_first_argument_is_larger():
    assert cohens_h(0.40, 0.30) > 0


def test_matches_known_closed_form():
    # h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2)), Cohen's standard definition
    p1, p2 = 0.5, 0.3
    expected = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    assert cohens_h(p1, p2) == pytest.approx(expected)
