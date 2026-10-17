import numpy as np
import pytest

from decision_triage import calibrate, metrics


def test_scale_identity_and_sharpening():
    p = np.array([[0.6, 0.3, 0.1]])
    assert calibrate.scale(p, 1.0) == pytest.approx(p)
    sharp = calibrate.scale(p, 0.5)
    assert sharp == pytest.approx(np.array([[0.36, 0.09, 0.01]]) / 0.46)
    assert calibrate.scale([[1.0, 0.0]], 1.0)[0, 1] == pytest.approx(1e-6, rel=1e-3)


@pytest.mark.parametrize("true_t", [2.0, 0.5])
def test_fit_recovers_known_temperature(softmax_data, true_t):
    probs, gold = softmax_data(20000, true_t)
    assert calibrate.fit_temperature(probs, gold) == pytest.approx(true_t, rel=0.05)


def test_fit_is_one_when_already_calibrated(softmax_data):
    probs, gold = softmax_data(20000, 1.0, seed=1)
    assert calibrate.fit_temperature(probs, gold) == pytest.approx(1.0, abs=0.05)


def test_fit_minimises_nll(softmax_data):
    probs, gold = softmax_data(2000, 2.0, seed=2)
    t = calibrate.fit_temperature(probs, gold)
    best = calibrate.nll(probs, gold, t)
    for other in (t * 0.9, t * 1.1, 1.0):
        assert calibrate.nll(probs, gold, other) > best


def test_calibrated_ece_beats_raw_on_underconfident_set(softmax_data):
    dev_p, dev_g = softmax_data(2000, 0.5, seed=3)
    test_p, test_g = softmax_data(5000, 0.5, seed=4)
    t = calibrate.fit_temperature(dev_p, dev_g)
    assert t < 1.0
    correct = test_p.argmax(axis=1) == test_g
    raw = metrics.ece(test_p.max(axis=1), correct)
    cal = metrics.ece(calibrate.scale(test_p, t).max(axis=1), correct)
    assert raw > 0.05
    assert cal < raw / 2
    assert np.array_equal(calibrate.scale(test_p, t).argmax(axis=1), test_p.argmax(axis=1))


def test_fit_empty_returns_identity():
    assert calibrate.fit_temperature(np.zeros((0, 7)), np.zeros(0, dtype=int)) == 1.0
