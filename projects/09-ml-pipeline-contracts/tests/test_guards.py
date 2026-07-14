import pandas as pd
import pytest

from guards import assert_no_target_leakage, filter_point_in_time


def test_filter_point_in_time_drops_post_decision_rows():
    events = pd.DataFrame({
        "customer_id": ["a", "a", "a", "b", "b"],
        "event_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01", "2024-01-05", "2024-03-01"]),
    })
    cutoffs = pd.Series({"a": pd.Timestamp("2024-01-15"), "b": pd.Timestamp("2024-01-15")})
    filtered = filter_point_in_time(events, cutoffs, "customer_id", "event_date")
    assert len(filtered) == 3  # a's 2024-02-01 row and b's 2024-03-01 row are both post-cutoff
    assert filtered["event_date"].max() <= pd.Timestamp("2024-01-15")


def test_filter_point_in_time_naive_use_would_leak_the_future():
    """The whole point of the guard: without it, a naive feature built
    from every row (not just pre-cutoff ones) sees events the decision
    couldn't have known about yet."""
    events = pd.DataFrame({
        "customer_id": ["a", "a"],
        "event_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),  # one legit, one from the far future
        "logins": [1, 99],
    })
    cutoffs = pd.Series({"a": pd.Timestamp("2024-01-15")})

    naive_avg = events["logins"].mean()
    filtered = filter_point_in_time(events, cutoffs, "customer_id", "event_date")
    correct_avg = filtered["logins"].mean()

    assert naive_avg == 50.0       # leaks the future event, badly skewed
    assert correct_avg == 1.0      # only the pre-cutoff event


def test_filter_point_in_time_missing_cutoff_raises():
    events = pd.DataFrame({"customer_id": ["z"], "event_date": pd.to_datetime(["2024-01-01"])})
    cutoffs = pd.Series({"a": pd.Timestamp("2024-01-15")})  # no entry for 'z'
    with pytest.raises(ValueError, match="no cutoff found"):
        filter_point_in_time(events, cutoffs, "customer_id", "event_date")


def test_assert_no_target_leakage_passes_when_clean():
    assert_no_target_leakage(["avg_logins", "plan_tier_basic"], ["churned"])  # no raise


def test_assert_no_target_leakage_catches_target_in_features():
    with pytest.raises(ValueError, match="target leakage"):
        assert_no_target_leakage(["avg_logins", "churned"], ["churned"])
