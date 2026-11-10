"""Stage 18 (Phase 6): predictor-source ablation, POWER vs IMDAA vs ERA5, 1990-2020.

LightGBM; train 1990-2009, calibrate 2010-2014, test 2015-2020. POWER/ERA5 features are
lagged one day; IMDAA features are aligned to the IMD day and need no lag (see stage 17).
Run A: all 34 stations (base, +POWER, +IMDAA, +both). Run B: stations with ERA5 (adds +ERA5,
+all three). Gain over "base" in Brier score, with year-month block-bootstrap 95% CIs.
Output: reports/results/ablation_3src.csv
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


def run(df: pd.DataFrame, sets: dict[str, list[str]], label: str) -> pd.DataFrame:
    yr = df["date"].dt.year
    tr, va, te = df[(yr >= 1990) & (yr <= 2009)], df[(yr >= 2010) & (yr <= 2014)], df[yr >= 2015].copy()
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
        rows.append({"run": label, "stations": df["station"].nunique(), "features": name,
                     "n_feats": len(sets[name]),
                     "bss_vs_clim": 1 - g["b_" + name].sum() / g["b_clim"].sum(),
                     "gain_over_base": (g["b_base"].sum() - g["b_" + name].sum()) / n.sum(),
                     "gain_lo": np.percentile(gain, 2.5), "gain_hi": np.percentile(gain, 97.5)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    i = pd.read_parquet(DATA_PROCESSED / "d2_daily.parquet")
    j = i.copy()
    j["date"] = j["date"] + pd.Timedelta(days=1)  # IMDAA of day t-1 seen at day t (same lag as POWER)
    j.columns = ["j_" + c[2:] if c.startswith("i_") else c for c in j.columns]
    df = f.merge(i, on=["station", "date"], how="inner").merge(j, on=["station", "date"], how="inner")
    cols = feature_columns(df)
    p_c = [c for c in cols if c.startswith("p_")]
    e_c = [c for c in cols if c.startswith("e_")]
    i_c = [c for c in cols if c.startswith("i_")]
    j_c = [c for c in cols if c.startswith("j_")]
    base = [c for c in cols if c not in p_c + e_c + i_c + j_c]
    res = [run(df, {"base": base, "base+POWER": base + p_c, "base+IMDAA": base + i_c,
                    "base+IMDAA_lag1": base + j_c, "base+POWER+IMDAA": base + p_c + i_c}, "A_all34")]
    have = df.groupby("station")["e_temperature_2m_mean"].apply(lambda s: s.notna().mean())
    sub = df[df["station"].isin(have[have > 0.9].index)]
    res.append(run(sub, {"base": base, "base+POWER": base + p_c, "base+IMDAA": base + i_c,
                         "base+IMDAA_lag1": base + j_c, "base+ERA5": base + e_c, "base+POWER+IMDAA+ERA5": base + p_c + i_c + e_c},
                   "B_era5_stations"))
    out = pd.concat(res).round(4)
    out.to_csv(OUT / "ablation_3src.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
