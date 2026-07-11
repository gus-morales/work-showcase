"""Data-generation invariants for all four synthetic datasets."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def golden():
    return gd.make_golden_eval_set(n=1200)


@pytest.fixture(scope="module")
def ab_results():
    return gd.make_ab_test_results(n=4000)


@pytest.fixture(scope="module")
def monitoring():
    return gd.make_quality_monitoring(n_days=120, regression_day=90, avg_tickets_per_day=80)


@pytest.fixture(scope="module")
def guardrail():
    return gd.make_guardrail_scores(n=4000)


# --- 1. Golden eval set --------------------------------------------------

def test_golden_labels_in_valid_range(golden):
    assert golden["human_label"].between(1, 5).all()
    assert golden["judge_label"].between(1, 5).all()


def test_golden_acceptable_flags_match_labels(golden):
    assert (golden["human_acceptable"] == (golden["human_label"] >= 4).astype(int)).all()
    assert (golden["judge_acceptable"] == (golden["judge_label"] >= 4).astype(int)).all()


def test_judge_is_more_generous_than_human_on_average(golden):
    assert golden["judge_label"].mean() > golden["human_label"].mean()


def test_complaint_category_has_largest_judge_bias(golden):
    bias = (golden["judge_label"] - golden["human_label"]).groupby(golden["category"]).mean()
    assert bias.idxmax() == "complaint"


# --- 2. A/B test results ---------------------------------------------

def test_ab_arms_roughly_balanced(ab_results):
    counts = ab_results["arm"].value_counts()
    assert abs(counts["v1_baseline"] - counts["v2_revised"]) < 0.1 * len(ab_results)


def test_v2_has_higher_acceptable_rate_than_v1(ab_results):
    rates = ab_results.groupby("arm")["judge_acceptable"].mean()
    assert rates["v2_revised"] > rates["v1_baseline"]


# --- 3. Quality monitoring ---------------------------------------------

def test_monitoring_covers_every_day(monitoring):
    assert sorted(monitoring["day"].tolist()) == list(range(120))


def test_monitoring_rate_drops_after_regression_day(monitoring):
    pre = monitoring[monitoring["day"] < 90]["acceptable_rate"].mean()
    post = monitoring[monitoring["day"] >= 90]["acceptable_rate"].mean()
    assert post < pre - 0.10


def test_n_acceptable_never_exceeds_n_tickets(monitoring):
    assert (monitoring["n_acceptable"] <= monitoring["n_tickets"]).all()


# --- 4. Guardrail scores ---------------------------------------------

def test_guardrail_risk_score_in_unit_interval(guardrail):
    assert guardrail["risk_score"].between(0, 1).all()


def test_guardrail_bad_replies_score_higher_on_average(guardrail):
    bad_mean = guardrail.loc[guardrail["true_bad"] == 1, "risk_score"].mean()
    good_mean = guardrail.loc[guardrail["true_bad"] == 0, "risk_score"].mean()
    assert bad_mean > good_mean


def test_guardrail_bad_rate_is_plausible(guardrail):
    rate = guardrail["true_bad"].mean()
    assert 0.03 < rate < 0.15
