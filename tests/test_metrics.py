"""Tests for forecast verification metrics.

The headline test is ``test_always_dry_scores_high_accuracy_but_zero_skill``:
it encodes the premise of the whole benchmark, namely that a model which never
forecasts rain looks good on accuracy and is worthless on every skill score.
"""

from __future__ import annotations

import numpy as np
import pytest

from indiarain.config import categorise_rainfall, station_by_name, zones
from indiarain.evaluation import metrics

# --------------------------------------------------------------------------- #
# Contingency table
# --------------------------------------------------------------------------- #


def test_contingency_counts():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    c = metrics.contingency(y_true, y_pred)
    assert (c.hits, c.false_alarms, c.misses, c.correct_negatives) == (2, 1, 1, 2)
    assert c.total == 6


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.contingency([1, 0, 1], [1, 0])


def test_non_binary_raises():
    with pytest.raises(ValueError, match="must be binary"):
        metrics.contingency([0, 1, 2], [0, 1, 1])


def test_empty_raises():
    with pytest.raises(ValueError, match="is empty"):
        metrics.contingency([], [])


# --------------------------------------------------------------------------- #
# The premise of the project
# --------------------------------------------------------------------------- #


def test_always_dry_scores_high_accuracy_but_zero_skill():
    """80% accuracy, zero skill: why this benchmark does not report accuracy.

    A forecaster that never predicts rain on a dataset with 20% rainy days
    gets 80% accuracy, but POD, CSI, HSS and ETS all collapse to 0 (or NaN
    where the score is undefined).
    """
    rng = np.random.default_rng(0)
    y_true = (rng.random(10_000) < 0.20).astype(int)
    y_pred = np.zeros_like(y_true)

    accuracy = float(np.mean(y_true == y_pred))
    assert accuracy == pytest.approx(0.80, abs=0.02), "sanity: ~80% accuracy"

    assert metrics.pod(y_true, y_pred) == 0.0
    assert metrics.csi(y_true, y_pred) == 0.0
    assert metrics.heidke_skill_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)
    assert metrics.equitable_threat_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)
    assert metrics.peirce_skill_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-9)
    # FAR is undefined with no rain forecasts at all.
    assert np.isnan(metrics.far(y_true, y_pred))


def test_perfect_forecast_scores_one():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    assert metrics.pod(y_true, y_true) == 1.0
    assert metrics.far(y_true, y_true) == 0.0
    assert metrics.csi(y_true, y_true) == 1.0
    assert metrics.heidke_skill_score(y_true, y_true) == pytest.approx(1.0)
    assert metrics.equitable_threat_score(y_true, y_true) == pytest.approx(1.0)
    assert metrics.frequency_bias(y_true, y_true) == pytest.approx(1.0)


def test_inverted_forecast_is_worse_than_chance():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    assert metrics.heidke_skill_score(y_true, 1 - y_true) < 0
    assert metrics.peirce_skill_score(y_true, 1 - y_true) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Probabilistic scores
# --------------------------------------------------------------------------- #


def test_brier_bounds():
    assert metrics.brier_score([1, 1, 0], [1.0, 1.0, 0.0]) == 0.0
    assert metrics.brier_score([1, 1, 0], [0.0, 0.0, 1.0]) == 1.0


def test_brier_rejects_out_of_range_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        metrics.brier_score([1, 0], [1.4, 0.2])


def test_brier_skill_score_zero_for_climatology():
    """Forecasting the base rate every day must score exactly 0 skill."""
    rng = np.random.default_rng(7)
    y_true = (rng.random(5_000) < 0.3).astype(int)
    climatology = np.full(y_true.shape, float(np.mean(y_true)))
    assert metrics.brier_skill_score(y_true, climatology) == pytest.approx(0.0, abs=1e-12)


def test_brier_decomposition_reconstructs_brier():
    """reliability - resolution + uncertainty == brier (Murphy 1973)."""
    rng = np.random.default_rng(42)
    y_prob = rng.random(4_000)
    y_true = (rng.random(4_000) < y_prob).astype(int)

    rel, res, unc = metrics.brier_decomposition(y_true, y_prob, n_bins=10)
    assert rel - res + unc == pytest.approx(metrics.brier_score(y_true, y_prob), abs=1e-3)


def test_well_calibrated_model_has_low_ece():
    """Probabilities generated as the true event rate should be calibrated."""
    rng = np.random.default_rng(1)
    y_prob = rng.random(50_000)
    y_true = (rng.random(50_000) < y_prob).astype(int)
    assert metrics.expected_calibration_error(y_true, y_prob) < 0.02


def test_overconfident_model_has_high_ece():
    """Pushing probabilities to the extremes must be penalised."""
    rng = np.random.default_rng(2)
    y_prob = rng.random(20_000)
    y_true = (rng.random(20_000) < y_prob).astype(int)
    overconfident = np.clip((y_prob - 0.5) * 4 + 0.5, 0, 1)

    assert metrics.expected_calibration_error(y_true, overconfident) > metrics.expected_calibration_error(
        y_true, y_prob
    )


def test_mce_at_least_ece():
    rng = np.random.default_rng(3)
    y_prob = rng.random(5_000)
    y_true = (rng.random(5_000) < y_prob).astype(int)
    assert metrics.maximum_calibration_error(y_true, y_prob) >= metrics.expected_calibration_error(
        y_true, y_prob
    )


# --------------------------------------------------------------------------- #
# Disagreement
# --------------------------------------------------------------------------- #


def test_identical_models_have_zero_disagreement():
    probs = np.tile(np.array([0.1, 0.5, 0.9]), (4, 1))
    assert np.allclose(metrics.disagreement_index(probs), 0.0)


def test_disagreement_is_per_sample():
    probs = np.array([[0.0, 0.5], [1.0, 0.5]])
    spread = metrics.disagreement_index(probs)
    assert spread.shape == (2,)
    assert spread[0] > spread[1] == 0.0


def test_disagreement_needs_two_models():
    with pytest.raises(ValueError, match="at least 2 models"):
        metrics.disagreement_index(np.array([[0.1, 0.2]]))


# --------------------------------------------------------------------------- #
# Summary row
# --------------------------------------------------------------------------- #


def test_summarise_returns_expected_keys():
    rng = np.random.default_rng(5)
    y_prob = rng.random(1_000)
    y_true = (rng.random(1_000) < y_prob).astype(int)

    row = metrics.summarise(y_true, y_prob)
    expected = {
        "pod", "far", "csi", "frequency_bias", "hss", "ets", "pss",
        "brier", "bss", "reliability", "resolution", "uncertainty", "ece", "mce",
    }
    assert set(row) == expected
    assert all(isinstance(v, float) for v in row.values())


# --------------------------------------------------------------------------- #
# IMD conventions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mm", "label"),
    [
        (0.0, "No rain"),
        (2.4, "No rain"),
        (2.5, "Light"),          # IMD rainy-day threshold is inclusive
        (15.5, "Light"),
        (15.6, "Moderate"),
        (64.4, "Moderate"),
        (64.5, "Heavy"),
        (115.6, "Very heavy"),
        (204.5, "Extremely heavy"),
        (900.0, "Extremely heavy"),
    ],
)
def test_imd_rainfall_categories(mm: float, label: str):
    assert categorise_rainfall(mm) == label


def test_negative_rainfall_raises():
    with pytest.raises(ValueError, match="cannot be negative"):
        categorise_rainfall(-1.0)


def test_station_lookup_is_case_insensitive():
    assert station_by_name("mumbai").state == "Maharashtra"
    with pytest.raises(KeyError):
        station_by_name("Atlantis")


def test_zones_are_unique_and_plural():
    z = zones()
    assert len(z) == len(set(z))
    assert len(z) >= 8, "leave-one-zone-out needs several distinct zones"
