"""Stage 14 (Phase 6, preliminary): predictor-source ablation on the stations that have ERA5.

Feature groups: rain history/calendar/indices ("base"), + NASA POWER ("p_"), + ERA5 ("e_").
LightGBM, lagged features, train 1981-2009, calibrate 2010-2014, test 2015-2024.
Skill = Brier skill vs per-station-month climatology, with year-month block-bootstrap CIs
for each set's gain over "base".
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


def main() -> int:
    df = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    have = df.groupby("station")["e_temperature_2m_mean"].apply(lambda s: s.notna().mean())
    stations = list(have[have > 0.9].index)
    df = df[df["station"].isin(stations)]
    yr = df["date"].dt.year
    tr, va, te = df[yr <= 2009], df[(yr >= 2010) & (yr <= 2014)], df[yr >= 2015].copy()
    cols = feature_columns(df)
    p_cols = [c for c in cols if c.startswith("p_")]
    e_cols = [c for c in cols if c.startswith("e_")]
    base = [c for c in cols if c not in p_cols and c not in e_cols]
    sets = {"base": base, "base+POWER": base + p_cols, "base+ERA5": base + e_cols,
            "base+POWER+ERA5": base + p_cols + e_cols}
    clim = tr.groupby(["station", "month"])["y"].mean().rename("c")
    te["b_clim"] = (te.join(clim, on=["station", "month"])["c"].fillna(tr["y"].mean()) - te["y"]) ** 2
    te["blk"] = te["date"].dt.year * 12 + te["date"].dt.month
    for name, fs in sets.items():
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                               n_jobs=8, verbose=-1, random_state=0).fit(tr[fs], tr["y"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(va[fs])[:, 1], va["y"])
        te["b_" + name] = (iso.predict(m.predict_proba(te[fs])[:, 1]) - te["y"]) ** 2
    g = te.groupby("blk")[[c for c in te.columns if c.startswith("b_") and c != "blk"]].sum()
    n = te.groupby("blk").size().to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(g), size=(1000, len(g)))
    tot = n[idx].sum(1)
    rows = []
    for name in sets:
        gain = (g["b_base"].to_numpy()[idx].sum(1) - g["b_" + name].to_numpy()[idx].sum(1)) / tot
        rows.append({"features": name, "n_feats": len(sets[name]),
                     "bss_vs_clim": 1 - g["b_" + name].sum() / g["b_clim"].sum(),
                     "gain_over_base_brier": (g["b_base"].sum() - g["b_" + name].sum()) / n.sum(),
                     "gain_lo": np.percentile(gain, 2.5), "gain_hi": np.percentile(gain, 97.5)})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / "ablation_sources_prelim.csv", index=False)
    print("stations:", stations)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
