"""Leak-safe feature construction for next-day rainfall.

Target: ``y_rain_next_mm`` (IMD rainfall of day t+1). The IMD day is the 24 h
ending 08:30 IST, so it straddles two calendar days (see PHASES.md, Phase 2).
Calendar-day predictors for day ``t`` overlap about 16 h of the target window.
Two feature sets are therefore built:

``lagged`` (main)    every reanalysis/POWER value is shifted one day, so only
                     information up to day t-1 is used; IMD rain up to day t is
                     allowed because it closes at 08:30 on day t.
``sameday`` (leaky)  reanalysis values of day t are used as they are. Reported
                     only as a sensitivity check on how much skill the overlap adds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indiarain.config import RAINY_DAY_MM, STATIONS

POWER_VARS = [
    "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "QV2M", "PS", "WS10M", "WD10M", "ALLSKY_SFC_SW_DWN",
]
INDEX_MONTHLY = ["nino34", "soi", "dmi"]
OMI = ["omi_pc1", "omi_pc2", "omi_amp"]


def _station_block(g: pd.DataFrame, mode: str) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    r = g["d1_rain_mm"]
    wet = (r >= RAINY_DAY_MM).astype(float).where(r.notna())
    out = pd.DataFrame(index=g.index)
    for k in (0, 1, 2, 3, 6):
        out[f"rain_lag{k}"] = r.shift(k)
    for w in (3, 7, 14, 30):
        out[f"rain_sum{w}"] = r.rolling(w, min_periods=max(2, w // 2)).sum()
        out[f"wet_frac{w}"] = wet.rolling(w, min_periods=max(2, w // 2)).mean()
    # days since last rainy day (capped), and current run length
    since = np.zeros(len(g))
    c = 60.0
    for i, v in enumerate(wet.to_numpy()):
        c = 0.0 if v == 1 else min(c + 1, 60.0)
        since[i] = c
    out["days_since_rain"] = since

    lag = 1 if mode == "lagged" else 0
    for v in POWER_VARS:
        s = g[f"d3_{v}"]
        out[f"p_{v}"] = s.shift(lag)
        out[f"p_{v}_d1"] = s.shift(lag) - s.shift(lag + 1)  # one-day change (tendency)
    out["p_rain"] = g["d3_PRECTOTCORR"].shift(1 if mode == "lagged" else 0)
    for col in g.columns:
        if col.startswith("d4_"):
            s = g[col]
            out["e_" + col[3:]] = s.shift(lag)
            if col != "d4_precipitation_sum":
                out["e_" + col[3:] + "_d1"] = s.shift(lag) - s.shift(lag + 1)
    for c_ in OMI:
        out[c_] = g[f"d5_{c_}"].shift(1)
    return out


def build(base: pd.DataFrame, mode: str = "lagged") -> pd.DataFrame:
    """Return the modelling table (one row per station-day) for ``mode``."""
    if mode not in {"lagged", "sameday"}:
        raise ValueError(mode)
    parts = []
    for st, g in base.groupby("station", sort=False):
        f = _station_block(g, mode)
        f["station"] = st
        f["date"] = g.sort_values("date")["date"].to_numpy()
        f["y_rain_next_mm"] = g.sort_values("date")["y_rain_next_mm"].to_numpy()
        # monthly indices: previous month only (published after month end)
        for c_ in INDEX_MONTHLY:
            f[c_] = g.sort_values("date")[f"d5_{c_}"].to_numpy()
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)

    # previous-month value of the monthly indices
    month = df["date"].values.astype("datetime64[M]")
    for c_ in INDEX_MONTHLY:
        tab = (
            pd.DataFrame({"m": month, "v": df[c_]}).dropna().drop_duplicates("m").set_index("m")["v"]
        )
        prev = pd.Series(df["date"].values.astype("datetime64[M]") - np.timedelta64(1, "M"))
        df[c_] = prev.map(tab).to_numpy()

    doy = df["date"].dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["date"].dt.month
    meta = {s.name: s for s in STATIONS}
    df["lat"] = df["station"].map(lambda n: meta[n].lat)
    df["lon"] = df["station"].map(lambda n: meta[n].lon)
    df["y"] = (df["y_rain_next_mm"] >= RAINY_DAY_MM).astype(float).where(df["y_rain_next_mm"].notna())
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"station", "date", "y_rain_next_mm", "y"}
    return [c for c in df.columns if c not in drop]
