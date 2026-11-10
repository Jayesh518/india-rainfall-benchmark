"""Stage 08 (Phases 4-5): baselines and first ML benchmark, temporal split.

train 1981-2009 | validation 2010-2014 (calibration, threshold) | test 2015-2024.
Writes reports/results/*.csv (git-ignored) and prints the headline table.
"""

from __future__ import annotations

import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from indiarain.config import DATA_PROCESSED, RAINY_DAY_MM, ROOT
from indiarain.evaluation import metrics as mt
from indiarain.features.build import feature_columns

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "results"
FLAGGED = ["Pune", "Shillong", "Coimbatore", "Leh"]  # unstable D1 labels (Phase 2)


def split(df: pd.DataFrame):
    yr = df["date"].dt.year
    return df[yr <= 2009], df[(yr >= 2010) & (yr <= 2014)], df[yr >= 2015]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.9, 35)
    return float(grid[int(np.argmax([mt.heidke_skill_score(y, p >= t) for t in grid]))])


def baselines(tr, va, te) -> dict[str, np.ndarray]:
    out = {"always_dry": np.zeros(len(te)), "persistence": (te["rain_lag0"] >= RAINY_DAY_MM) * 1.0}
    clim = tr.groupby(["station", "month"])["y"].mean().rename("c")
    out["climatology"] = te.join(clim, on=["station", "month"])["c"].fillna(tr["y"].mean()).to_numpy()
    trw = tr.assign(w=(tr["rain_lag0"] >= RAINY_DAY_MM).astype(int))
    mk = trw.groupby(["station", "month", "w"])["y"].mean().rename("m")
    tew = te.assign(w=(te["rain_lag0"] >= RAINY_DAY_MM).astype(int))
    out["markov1"] = tew.join(mk, on=["station", "month", "w"])["m"].fillna(tr["y"].mean()).to_numpy()
    return out


def models(cols):
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=300, C=0.5)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(300, min_samples_leaf=20,
                                                              n_jobs=8, random_state=0)),
        "hist_gb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                  early_stopping=True, random_state=0),
        "lightgbm": lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       min_child_samples=50, n_jobs=8, verbose=-1, random_state=0),
    }


def run(mode: str, drop_flagged: bool):
    df = pd.read_parquet(DATA_PROCESSED / f"features_{mode}.parquet")
    if drop_flagged:
        df = df[~df["station"].isin(FLAGGED)]
    tr, va, te = split(df)
    cols = feature_columns(df)
    rows, preds = [], {}
    for name, p in baselines(tr, va, te).items():
        thr = 0.5 if name in {"always_dry", "persistence"} else best_threshold(
            va["y"].to_numpy(), baselines(tr, va, va)[name])
        preds[name] = (p, thr)
    for name, mdl in models(cols).items():
        mdl.fit(tr[cols], tr["y"])
        pv, pt = mdl.predict_proba(va[cols])[:, 1], mdl.predict_proba(te[cols])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, va["y"])
        pt_cal = iso.predict(pt)
        preds[name] = (pt_cal, best_threshold(va["y"].to_numpy(), iso.predict(pv)))
    yt = te["y"].to_numpy()
    clim_bs = mt.brier_score(yt, preds["climatology"][0])
    for name, (p, thr) in preds.items():
        s = mt.summarise(yt, p, threshold=thr)
        s.update(model=name, mode=mode, drop_flagged=drop_flagged, thr=round(thr, 2),
                 bss_vs_clim=1 - mt.brier_score(yt, p) / clim_bs)
        rows.append(s)
    per = []
    for name in ("climatology", "lightgbm"):
        p, thr = preds[name]
        for st, idx in te.reset_index(drop=True).groupby("station").groups.items():
            ys, ps = yt[list(idx)], p[list(idx)]
            per.append({"model": name, "station": st, "mode": mode, "n": len(ys),
                        "brier": mt.brier_score(ys, ps),
                        "hss": mt.heidke_skill_score(ys, ps >= thr),
                        "csi": mt.csi(ys, ps >= thr)})
    return pd.DataFrame(rows), pd.DataFrame(per), te, preds


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res, per = [], []
    for mode in ("lagged", "sameday"):
        for drop in (False, True):
            r, p, _, _ = run(mode, drop)
            res.append(r)
            per.append(p.assign(drop_flagged=drop))
            print(f"done {mode} drop_flagged={drop}", flush=True)
    res = pd.concat(res)
    res.to_csv(OUT / "benchmark_v1.csv", index=False)
    pd.concat(per).to_csv(OUT / "per_station_v1.csv", index=False)
    show = ["model", "mode", "drop_flagged", "thr", "brier", "bss_vs_clim", "hss", "csi", "pod", "far", "ece"]
    show = [c for c in show if c in res.columns]
    print(res[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
