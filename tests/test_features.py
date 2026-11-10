"""Leakage tests for the feature builder: row t must not see day t+1 or (lagged mode) day t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indiarain.features.build import POWER_VARS, build, feature_columns


def _base(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n), "station": "Delhi"})
    d["d1_rain_mm"] = rng.gamma(0.3, 8, n)
    d["y_rain_next_mm"] = d["d1_rain_mm"].shift(-1)
    for v in [*POWER_VARS, "PRECTOTCORR"]:
        d[f"d3_{v}"] = rng.normal(size=n)
    for c in ("nino34", "soi", "dmi", "omi_pc1", "omi_pc2", "omi_amp"):
        d[f"d5_{c}"] = rng.normal(size=n)
    return d


def _row(df: pd.DataFrame, i: int) -> pd.Series:
    return df.sort_values("date").iloc[i]


def test_lagged_features_ignore_same_day_reanalysis():
    base = _base()
    a = build(base, "lagged")
    b_base = base.copy()
    t = 60
    for v in [*POWER_VARS, "PRECTOTCORR"]:
        b_base.loc[t, f"d3_{v}"] += 100.0
    b = build(b_base, "lagged")
    cols = feature_columns(a)
    pa, pb = a.sort_values("date").iloc[t][cols], b.sort_values("date").iloc[t][cols]
    p_cols = [c for c in cols if c.startswith("p_")]
    assert np.allclose(pa[p_cols].astype(float), pb[p_cols].astype(float), equal_nan=True)


def test_sameday_features_do_see_same_day_reanalysis():
    base = _base()
    b_base = base.copy()
    b_base.loc[60, "d3_T2M"] += 100.0
    a, b = build(base, "sameday"), build(b_base, "sameday")
    assert a.sort_values("date").iloc[60]["p_T2M"] != b.sort_values("date").iloc[60]["p_T2M"]


def test_no_feature_depends_on_future_rain():
    base = _base()
    cut = 70
    mod = base.copy()
    mod.loc[cut + 1 :, "d1_rain_mm"] = 999.0
    mod["y_rain_next_mm"] = mod["d1_rain_mm"].shift(-1)
    a, b = build(base, "lagged"), build(mod, "lagged")
    cols = feature_columns(a)
    ra, rb = a.sort_values("date").iloc[cut][cols], b.sort_values("date").iloc[cut][cols]
    assert np.allclose(ra.astype(float), rb.astype(float), equal_nan=True)


def test_target_is_next_day_binary():
    base = _base()
    df = build(base, "lagged").sort_values("date")
    expect = (base["d1_rain_mm"].shift(-1) >= 2.5).astype(float).where(base["d1_rain_mm"].shift(-1).notna())
    assert np.array_equal(df["y"].to_numpy(), expect.to_numpy(), equal_nan=True)
