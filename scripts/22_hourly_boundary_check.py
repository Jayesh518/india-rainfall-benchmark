"""Stage 22: boundary-sensitivity (leak) check for hourly-aligned POWER features.

The aligned window is moved by s hours (positive = earlier) (s = -1, 0, 1, 3, 6, 12; s = -1 deliberately LEAKS one hour as a positive control): stamps
(04Z-s of d-1 .. 03Z-s of d). If skill came from a stamp leaking into the target window it
would collapse as soon as s >= 1; genuine recency information decays smoothly with s.
Output: reports/results/hourly_boundary_check.csv
"""

from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from indiarain.config import DATA_PROCESSED, DATA_RAW, ROOT
from indiarain.features.build import feature_columns

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "results"


def aligned(h: pd.DataFrame, s: int) -> pd.DataFrame:
    t = h["time"] - pd.Timedelta(hours=4 - s)  # s > 0 moves the window EARLIER
    h = h.assign(date=t.dt.floor("D") + pd.Timedelta(days=1), hr=t.dt.hour)
    g = h.groupby(["station", "date"])
    n = g["T2M"].count()
    out = pd.DataFrame({
        "rain_sum": g["PRECTOTCORR"].sum(), "rain_max1h": g["PRECTOTCORR"].max(),
        "T2M_mean": g["T2M"].mean(), "RH_mean": g["RH2M"].mean(), "QV_mean": g["QV2M"].mean(),
        "PS_mean": g["PS"].mean(), "PS_d": g["PS"].last() - g["PS"].first(),
        "WS_mean": g["WS10M"].mean(), "SW_sum": g["ALLSKY_SFC_SW_DWN"].sum(),
    })
    last = h[h["hr"] >= 18].groupby(["station", "date"])
    out["rain_last6"] = last["PRECTOTCORR"].sum()
    out["RH_last6"] = last["RH2M"].mean()
    out["QV_last6"] = last["QV2M"].mean()
    return out[n == 24].add_prefix("h_").reset_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in sorted((DATA_RAW / "d3_power_hourly").glob("*.parquet")):
        d = pd.read_parquet(f).reset_index()
        d["station"] = f.stem.rsplit("_", 1)[0].replace("_", " ")
        frames.append(d)
    h = pd.concat(frames, ignore_index=True)
    f = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    cols0 = feature_columns(f)
    base = [c for c in cols0 if not c.startswith(("p_", "e_"))]
    rows = []
    for s in (-1, 0, 1, 3, 6, 12):
        df = f.merge(aligned(h, s), on=["station", "date"], how="inner")
        yr = df["date"].dt.year
        tr, va, te = df[(yr >= 2001) & (yr <= 2009)], df[(yr >= 2010) & (yr <= 2014)], df[yr >= 2015]
        hc = [c for c in df.columns if c.startswith("h_")]
        clim = tr.groupby(["station", "month"])["y"].mean().rename("c")
        bclim = ((te.join(clim, on=["station", "month"])["c"].fillna(tr["y"].mean()) - te["y"]) ** 2).mean()
        fs = base + hc
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                               n_jobs=8, verbose=-1, random_state=0).fit(tr[fs], tr["y"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(va[fs])[:, 1], va["y"])
        b = ((iso.predict(m.predict_proba(te[fs])[:, 1]) - te["y"]) ** 2).mean()
        rows.append({"window_shifted_earlier_h": s, "bss_vs_clim": 1 - b / bclim})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "hourly_boundary_check.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
