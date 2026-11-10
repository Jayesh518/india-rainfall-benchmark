"""Stage 21 (Phase 6): does cutting the day at the IMD boundary recover the lost skill?

Feature sets (LightGBM, train 2001-2009, calibrate 2010-2014, test 2015-2024 unless noted):
  base                   rain history, calendar, indices, static
  +POWER daily lag1      calendar-day POWER shifted one day (the main-benchmark design)
  +POWER hourly aligned  hourly POWER cut at 03Z, no lag (h_ features)
  +POWER hourly lag1     same hourly features shifted one day (control: isolates the timing effect)
  +both
A second run restricted to 2015-2020 adds IMDAA aligned for reference.
Output: reports/results/ablation_hourly.csv
"""

from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from indiarain.config import DATA_PROCESSED, ROOT
from indiarain.features.build import feature_columns

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "results"


def run(df, sets, label, last_year):
    yr = df["date"].dt.year
    tr = df[(yr >= 2001) & (yr <= 2009)]
    va = df[(yr >= 2010) & (yr <= 2014)]
    te = df[(yr >= 2015) & (yr <= last_year)].copy()
    clim = tr.groupby(["station", "month"])["y"].mean().rename("c")
    te["b_clim"] = (te.join(clim, on=["station", "month"])["c"].fillna(tr["y"].mean()) - te["y"]) ** 2
    te["blk"] = te["date"].dt.year * 12 + te["date"].dt.month
    for name, fs in sets.items():
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                               n_jobs=8, verbose=-1, random_state=0).fit(tr[fs], tr["y"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(va[fs])[:, 1], va["y"])
        te["b_" + name] = (iso.predict(m.predict_proba(te[fs])[:, 1]) - te["y"]) ** 2
    g = te.groupby("blk")[[c for c in te.columns if c.startswith("b_")]].sum()
    n = te.groupby("blk").size().to_numpy()
    idx = np.random.default_rng(0).integers(0, len(g), size=(1000, len(g)))
    tot = n[idx].sum(1)
    rows = []
    for name in sets:
        gain = (g["b_base"].to_numpy()[idx].sum(1) - g["b_" + name].to_numpy()[idx].sum(1)) / tot
        rows.append({"run": label, "features": name, "n_feats": len(sets[name]),
                     "bss_vs_clim": 1 - g["b_" + name].sum() / g["b_clim"].sum(),
                     "gain_over_base": (g["b_base"].sum() - g["b_" + name].sum()) / n.sum(),
                     "gain_lo": np.percentile(gain, 2.5), "gain_hi": np.percentile(gain, 97.5)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    h = pd.read_parquet(DATA_PROCESSED / "power_hourly_daily.parquet")
    h1 = h.copy()
    h1["date"] = h1["date"] + pd.Timedelta(days=1)
    h1.columns = ["k_" + c[2:] if c.startswith("h_") else c for c in h1.columns]
    df = f.merge(h, on=["station", "date"], how="inner").merge(h1, on=["station", "date"], how="inner")
    cols = feature_columns(df)
    p_c = [c for c in cols if c.startswith("p_")]
    h_c = [c for c in cols if c.startswith("h_")]
    k_c = [c for c in cols if c.startswith("k_")]
    e_c = [c for c in cols if c.startswith("e_")]
    base = [c for c in cols if c not in p_c + h_c + k_c + e_c]
    sets = {"base": base, "base+POWER_daily_lag1": base + p_c,
            "base+POWER_hourly_lag1": base + k_c, "base+POWER_hourly_aligned": base + h_c,
            "base+POWER_daily_lag1+hourly_aligned": base + p_c + h_c}
    res = [run(df, sets, "2015-2024", 2024)]
    i = pd.read_parquet(DATA_PROCESSED / "d2_daily.parquet")
    d2 = df.merge(i, on=["station", "date"], how="inner")
    i_c = [c for c in d2.columns if c.startswith("i_")]
    sets2 = {**sets, "base+IMDAA_aligned": base + i_c}
    res.append(run(d2, sets2, "2015-2020_with_IMDAA", 2020))
    out = pd.concat(res).round(4)
    out.to_csv(OUT / "ablation_hourly.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
