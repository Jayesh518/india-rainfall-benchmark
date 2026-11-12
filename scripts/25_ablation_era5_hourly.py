"""Stage 25: hourly ERA5 aligned to the IMD day vs hourly POWER and daily ERA5, same stations.

Output: reports/results/ablation_era5_hourly.csv
"""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd

from indiarain.config import DATA_PROCESSED, ROOT
from indiarain.features.build import feature_columns

spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "21_ablation_hourly.py")
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)


def main() -> int:
    f = pd.read_parquet(DATA_PROCESSED / "features_lagged.parquet")
    h = pd.read_parquet(DATA_PROCESSED / "power_hourly_daily.parquet")
    w = pd.read_parquet(DATA_PROCESSED / "era5_hourly_daily.parquet")
    w1 = w.copy()
    w1["date"] = w1["date"] + pd.Timedelta(days=1)
    w1.columns = ["x_" + c[2:] if c.startswith("w_") else c for c in w1.columns]
    df = f[f["station"].isin(w["station"].unique())]
    df = df.merge(h, on=["station", "date"]).merge(w, on=["station", "date"]).merge(
        w1, on=["station", "date"])
    cols = feature_columns(df)
    grp = {p: [c for c in cols if c.startswith(p)] for p in ("p_", "h_", "w_", "x_", "e_")}
    skip = [c for v in grp.values() for c in v]
    base = [c for c in cols if c not in skip]
    sets = {"base": base, "base+POWER_hourly_aligned": base + grp["h_"],
            "base+ERA5_hourly_lag1": base + grp["x_"],
            "base+ERA5_hourly_aligned": base + grp["w_"],
            "base+POWER_aligned+ERA5_aligned": base + grp["h_"] + grp["w_"]}
    if grp["e_"]:
        sets["base+ERA5_daily_lag1"] = base + grp["e_"]
    out = abl.run(df, sets, f"{df['station'].nunique()}_stations_2015-2024", 2024).round(4)
    out.to_csv(ROOT / "reports" / "results" / "ablation_era5_hourly.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
