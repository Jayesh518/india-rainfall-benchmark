"""Stage 09 (Phases 5/7): uncertainty, zone breakdown and feature importance.

Trains LightGBM on the lagged table (all stations), then reports on the 2015-2024 test set:
block-bootstrap 95% CIs (blocks = year-month, resampled with replacement) for the Brier
skill score against climatology and the Markov chain, BSS per zone, and gain importance.
"""

from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from indiarain.config import DATA_PROCESSED, ROOT, STATIONS
from indiarain.features.build import feature_columns

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "results"
RNG = np.random.default_rng(0)
N_BOOT = 1000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    yr = df["date"].dt.year
    tr, va, te = df[yr <= 2009], df[(yr >= 2010) & (yr <= 2014)], df[yr >= 2015].copy()
    cols = feature_columns(df)
    mdl = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                             n_jobs=8, verbose=-1, random_state=0).fit(tr[cols], tr["y"])
    iso = IsotonicRegression(out_of_bounds="clip").fit(mdl.predict_proba(va[cols])[:, 1], va["y"])
    te["p_lgb"] = iso.predict(mdl.predict_proba(te[cols])[:, 1])
    clim = tr.groupby(["station", "month"])["y"].mean().rename("c")
    te["p_clim"] = te.join(clim, on=["station", "month"])["c"].fillna(tr["y"].mean())
    w = (tr["rain_lag0"] >= 2.5).astype(int)
    mk = tr.assign(w=w).groupby(["station", "month", "w"])["y"].mean().rename("m")
    tw = te.assign(w=(te["rain_lag0"] >= 2.5).astype(int))
    te["p_mk"] = tw.join(mk, on=["station", "month", "w"])["m"].fillna(tr["y"].mean())

    te["blk"] = te["date"].dt.year * 12 + te["date"].dt.month
    te["b_lgb"] = (te["p_lgb"] - te["y"]) ** 2
    te["b_clim"] = (te["p_clim"] - te["y"]) ** 2
    te["b_mk"] = (te["p_mk"] - te["y"]) ** 2
    g = te.groupby("blk")[["b_lgb", "b_clim", "b_mk"]].agg(["sum"])
    g.columns = ["lgb", "clim", "mk"]
    n = te.groupby("blk").size().to_numpy()
    arr = g.to_numpy()
    idx = RNG.integers(0, len(g), size=(N_BOOT, len(g)))
    tot = n[idx].sum(1)
    bs = {k: arr[idx, i].sum(1) / tot for i, k in enumerate(g.columns)}
    res = []
    for name, ref in (("lgb_vs_clim", "clim"), ("lgb_vs_markov", "mk"), ("markov_vs_clim", "clim")):
        a = "mk" if name.startswith("markov") else "lgb"
        b = bs[ref]
        skill = 1 - bs[a] / b
        res.append({"comparison": name, "bss": 1 - arr[:, list(g.columns).index(a)].sum()
                    / arr[:, list(g.columns).index(ref)].sum(),
                    "ci_lo": np.percentile(skill, 2.5), "ci_hi": np.percentile(skill, 97.5)})
    ci = pd.DataFrame(res).round(4)
    ci.to_csv(OUT / "bss_bootstrap_ci.csv", index=False)

    zone = {s.name: s.zone for s in STATIONS}
    te["zone"] = te["station"].map(zone)
    z = te.groupby("zone").apply(
        lambda d: pd.Series({"n": len(d), "rain_rate": d["y"].mean(),
                             "bss_lgb_vs_clim": 1 - d["b_lgb"].mean() / d["b_clim"].mean()}))
    z.round(3).to_csv(OUT / "bss_by_zone.csv")
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    imp.to_csv(OUT / "lightgbm_gain_importance.csv", header=["share"])
    print(ci.to_string(index=False))
    print("\nBSS by zone:\n", z.round(3).sort_values("bss_lgb_vs_clim").to_string())
    print("\nTop features (gain share):\n", imp.head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
