"""Forecast verification metrics: meteorological scores and calibration.

Accuracy is close to useless for rainfall. With ~80% dry days, a model that
always answers "no rain" scores 80% and looks respectable. The scores here are
the ones the meteorological community actually uses, plus the calibration
measures that reveal whether a stated confidence means anything.

All functions take binary ``y_true`` and either binary ``y_pred`` or
probabilistic ``y_prob``, and return plain floats so results tabulate cleanly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


class ContingencyTable(NamedTuple):
    """2x2 verification table.

    Named per meteorological convention: a forecast "event" is rain.

    hits    (a) - forecast rain, observed rain
    misses  (c) - forecast dry,  observed rain
    false_alarms (b) - forecast rain, observed dry
    correct_negatives (d) - forecast dry, observed dry
    """

    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int

    @property
    def total(self) -> int:
        return self.hits + self.false_alarms + self.misses + self.correct_negatives


def _as_binary(y: ArrayLike, name: str) -> npt.NDArray[np.int_]:
    arr = np.asarray(y).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    uniq = set(np.unique(arr).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"{name} must be binary 0/1, got values {sorted(uniq)[:5]}")
    return arr.astype(int)


def contingency(y_true: ArrayLike, y_pred: ArrayLike) -> ContingencyTable:
    """Build the 2x2 verification table."""
    t = _as_binary(y_true, "y_true")
    p = _as_binary(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_pred {p.shape}")
    return ContingencyTable(
        hits=int(np.sum((p == 1) & (t == 1))),
        false_alarms=int(np.sum((p == 1) & (t == 0))),
        misses=int(np.sum((p == 0) & (t == 1))),
        correct_negatives=int(np.sum((p == 0) & (t == 0))),
    )


def _safe_div(num: float, den: float) -> float:
    """Return NaN rather than raising when a score is undefined."""
    return float(num) / float(den) if den else float("nan")


# --------------------------------------------------------------------------- #
# Categorical (deterministic) scores
# --------------------------------------------------------------------------- #


def pod(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Probability of Detection (hit rate, recall). Perfect = 1."""
    c = contingency(y_true, y_pred)
    return _safe_div(c.hits, c.hits + c.misses)


def far(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """False Alarm Ratio: fraction of rain forecasts that were wrong. Perfect = 0."""
    c = contingency(y_true, y_pred)
    return _safe_div(c.false_alarms, c.hits + c.false_alarms)


def csi(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Critical Success Index (threat score). Ignores correct negatives, so
    unlike accuracy it cannot be inflated by the abundance of dry days.
    Perfect = 1."""
    c = contingency(y_true, y_pred)
    return _safe_div(c.hits, c.hits + c.misses + c.false_alarms)


def frequency_bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Ratio of forecast to observed event frequency. >1 over-forecasts rain."""
    c = contingency(y_true, y_pred)
    return _safe_div(c.hits + c.false_alarms, c.hits + c.misses)


def heidke_skill_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Heidke Skill Score: accuracy relative to random chance.

    0 means no better than chance, 1 is perfect, negative is worse than chance.
    This is the score that exposes a "always predict dry" model: it gets
    HSS = 0 despite ~80% accuracy.
    """
    c = contingency(y_true, y_pred)
    a, b, cc, d = c.hits, c.false_alarms, c.misses, c.correct_negatives
    n = c.total
    expected = ((a + b) * (a + cc) + (cc + d) * (b + d)) / n
    return _safe_div(a + d - expected, n - expected)


def equitable_threat_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Equitable Threat Score (Gilbert Skill Score): CSI adjusted for the hits
    expected by chance. Range -1/3 to 1; 0 is no skill."""
    c = contingency(y_true, y_pred)
    a, b, cc = c.hits, c.false_alarms, c.misses
    hits_random = (a + b) * (a + cc) / c.total
    return _safe_div(a - hits_random, a + b + cc - hits_random)


def peirce_skill_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Peirce Skill Score (Hanssen-Kuipers): POD minus false alarm rate."""
    c = contingency(y_true, y_pred)
    hit_rate = _safe_div(c.hits, c.hits + c.misses)
    false_rate = _safe_div(c.false_alarms, c.false_alarms + c.correct_negatives)
    return hit_rate - false_rate


# --------------------------------------------------------------------------- #
# Probabilistic scores
# --------------------------------------------------------------------------- #


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better."""
    t = _as_binary(y_true, "y_true").astype(float)
    p = np.asarray(y_prob, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_prob {p.shape}")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("y_prob must lie in [0, 1]")
    return float(np.mean((p - t) ** 2))


def brier_skill_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Brier score relative to a climatology forecast (the observed base rate).

    0 means no better than always forecasting the long-run base rate -- the
    benchmark most published rainfall models are never compared against.
    """
    t = _as_binary(y_true, "y_true").astype(float)
    reference = np.full_like(t, float(np.mean(t)))
    bs = brier_score(t, y_prob)
    bs_ref = brier_score(t, reference)
    return _safe_div(bs_ref - bs, bs_ref)


def brier_decomposition(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> tuple[float, float, float]:
    """Murphy's decomposition: ``(reliability, resolution, uncertainty)``.

    ``brier = reliability - resolution + uncertainty``. Reliability is the
    calibration penalty (lower is better); resolution is the ability to
    discriminate (higher is better); uncertainty depends only on the data.
    """
    t = _as_binary(y_true, "y_true").astype(float)
    p = np.asarray(y_prob, dtype=float).ravel()
    base = float(np.mean(t))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    reliability = resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        mean_p = float(np.mean(p[mask]))
        mean_o = float(np.mean(t[mask]))
        reliability += n_b * (mean_p - mean_o) ** 2
        resolution += n_b * (mean_o - base) ** 2

    n = t.size
    return reliability / n, resolution / n, base * (1.0 - base)


def expected_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 15
) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| over bins,
    weighted by bin population. 0 is perfectly calibrated.

    This is the number that answers "when this model says 70%, does it rain
    70% of the time?"
    """
    t = _as_binary(y_true, "y_true").astype(float)
    p = np.asarray(y_prob, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_prob {p.shape}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        ece += (n_b / t.size) * abs(float(np.mean(p[mask])) - float(np.mean(t[mask])))
    return ece


def maximum_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 15
) -> float:
    """Worst-case calibration gap across populated bins."""
    t = _as_binary(y_true, "y_true").astype(float)
    p = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    gaps = [
        abs(float(np.mean(p[idx == b])) - float(np.mean(t[idx == b])))
        for b in range(n_bins)
        if np.any(idx == b)
    ]
    return max(gaps) if gaps else float("nan")


# --------------------------------------------------------------------------- #
# Model disagreement
# --------------------------------------------------------------------------- #


def disagreement_index(probs: ArrayLike) -> npt.NDArray[np.float64]:
    """Per-sample standard deviation across an ensemble's predicted probabilities.

    ``probs`` is ``(n_models, n_samples)``. High spread marks days the models
    cannot agree on; whether that spread predicts error is RQ1, and it is the
    state signal the RL ensemble-selection agent consumes.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_models, n_samples), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("need at least 2 models to measure disagreement")
    return np.std(arr, axis=0, ddof=0)


def summarise(y_true: ArrayLike, y_prob: ArrayLike, threshold: float = 0.5) -> dict[str, float]:
    """Full metric row for one model, ready for ``results/metrics.csv``."""
    y_pred = (np.asarray(y_prob, dtype=float).ravel() >= threshold).astype(int)
    reliability, resolution, uncertainty = brier_decomposition(y_true, y_prob)
    return {
        "pod": pod(y_true, y_pred),
        "far": far(y_true, y_pred),
        "csi": csi(y_true, y_pred),
        "frequency_bias": frequency_bias(y_true, y_pred),
        "hss": heidke_skill_score(y_true, y_pred),
        "ets": equitable_threat_score(y_true, y_pred),
        "pss": peirce_skill_score(y_true, y_pred),
        "brier": brier_score(y_true, y_prob),
        "bss": brier_skill_score(y_true, y_prob),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "ece": expected_calibration_error(y_true, y_prob),
        "mce": maximum_calibration_error(y_true, y_prob),
    }
