import math

import numpy as np
import pytest

from decision_triage import metrics


def test_ece_hand_computed():
    conf = [0.9, 0.9, 0.9, 0.9, 0.62, 0.62]
    correct = [1, 1, 1, 0, 1, 0]
    # bin 13: acc 0.75, conf 0.90, weight 4/6 -> 0.10 ; bin 9: acc 0.5, conf 0.62, weight 2/6 -> 0.04
    assert metrics.ece(conf, correct) == pytest.approx(0.14)
    bins = metrics.reliability_bins(conf, correct)
    assert len(bins) == 15
    assert bins[13].count == 4 and bins[13].accuracy == 0.75
    assert bins[9].count == 2 and bins[9].confidence == pytest.approx(0.62)
    assert sum(b.count for b in bins) == 6


def test_perfect_calibration():
    assert metrics.ece([0.8] * 10, [1] * 8 + [0] * 2) == pytest.approx(0.0)


def test_overconfident_near_miss():
    assert metrics.ece([0.99] * 10, [1] * 9 + [0]) == pytest.approx(0.09)


def test_confidence_one_and_zero_land_in_edge_bins():
    bins = metrics.reliability_bins([0.0, 1.0], [0, 1])
    assert bins[0].count == 1 and bins[-1].count == 1


def test_accuracy_and_empty():
    assert metrics.accuracy([1, 0, 1, 1]) == 0.75
    assert math.isnan(metrics.accuracy([]))
    assert math.isnan(metrics.ece([], []))


def test_coverage_curve_monotonic_and_edges():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    correct = rng.random(200) < conf
    t, cov, acc = metrics.coverage_curve(conf, correct)
    assert cov[0] == 1.0
    assert np.all(np.diff(cov) <= 0)
    assert np.all((acc[cov > 0] >= 0) & (acc[cov > 0] <= 1))
    assert np.all(np.isnan(acc[cov == 0]))

    _, cov, acc = metrics.coverage_curve([1.0, 1.0], [1, 0])
    assert cov[-1] == 1.0 and acc[-1] == 0.5

    _, cov, acc = metrics.coverage_curve([0.5, 0.5, 0.5], [1, 1, 0])
    assert set(np.unique(cov)) == {0.0, 1.0}

    _, cov, acc = metrics.coverage_curve([0.9, 0.9], [1, 1], refused=[True, True])
    assert np.all(cov == 0) and np.all(np.isnan(acc))

    _, cov, _ = metrics.coverage_curve([], [])
    assert np.all(cov == 0)


def test_refusals_never_covered_but_count_in_denominator():
    _, cov, acc = metrics.coverage_curve([0.9, 0.0], [1, 0], refused=[False, True])
    assert cov[0] == 0.5 and acc[0] == 1.0


def test_choose_threshold_toy():
    conf = [0.99, 0.97, 0.9, 0.8, 0.6]
    correct = [1, 1, 1, 0, 1]
    assert metrics.choose_threshold(conf, correct, target=0.98) == pytest.approx(0.801)
    assert metrics.choose_threshold(conf, correct, target=0.8) == 0.0
    assert metrics.choose_threshold([0.9, 0.9], [0, 0], target=0.98) is None
    cov, acc = metrics.at_threshold(0.801, conf, correct)
    assert cov == pytest.approx(0.6) and acc == 1.0


def test_aurc_hand_example():
    # ranked: ok, wrong, ok, ok -> risks 0, 1/2, 1/3, 1/4
    assert metrics.aurc([0.6, 0.9, 0.7, 0.8], [1, 1, 1, 0]) == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    assert metrics.aurc([0.9, 0.5], [1, 1]) == 0.0
    assert metrics.aurc([0.9, 0.5], [0, 0]) == 1.0
    assert math.isnan(metrics.aurc([], []))


def test_aurc_ties_use_expected_order():
    # tie of {ok, wrong}: k=1 expects 0.5 errors -> risk 0.5; k=2 -> 0.5
    assert metrics.aurc([0.9, 0.9], [1, 0]) == pytest.approx(0.5)
    assert metrics.aurc([0.9, 0.9], [0, 1]) == pytest.approx(0.5)
    # ok, then tie {ok, wrong}: risks 0, 0.25, 1/3
    assert metrics.aurc([0.95, 0.9, 0.9], [1, 0, 1]) == pytest.approx((0 + 0.25 + 1 / 3) / 3)


def test_aurc_better_ranking_scores_lower():
    correct = [1, 1, 1, 0, 0]
    good = metrics.aurc([0.9, 0.8, 0.7, 0.6, 0.5], correct)
    bad = metrics.aurc([0.5, 0.6, 0.7, 0.8, 0.9], correct)
    assert good < bad


def test_latency_and_cost():
    p50, p95 = metrics.latency_percentiles(list(range(1, 101)))
    assert p50 == pytest.approx(50.5) and p95 == pytest.approx(95.05)
    assert metrics.cost_per_1k([0.001, 0.003]) == pytest.approx(2.0)
