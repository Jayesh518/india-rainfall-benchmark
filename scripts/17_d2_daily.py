"""Stage 17: IMDAA daily features aligned to the IMD day.

IMD day d = 24 h ending 03:00 UTC (08:30 IST) on d. The window (03Z(d-1), 03Z(d)] holds the
stamps 06Z, 12Z, 18Z of d-1 and 00Z of d. Everything in it is observed before the next IMD day
begins, so these features need no one-day lag when predicting day d+1 (Phase 2 finding).
Output: data/processed/d2_daily.parquet (station, date, i_* features).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from indiarain.config import DATA_INTERIM, DATA_PROCESSED


def main() -> int:
    d2 = pd.read_parquet(DATA_INTERIM / "d2_station_6h.parquet")
    d2["date"] = (d2["time"] - pd.Timedelta(hours=3, seconds=1)).dt.floor("D") + pd.Timedelta(days=1)
    d2["wspd10"] = np.hypot(d2["U10"], d2["V10"])
    d2["wspd850"] = np.hypot(d2["UGRD850"], d2["VGRD850"])
    d2["wspd200"] = np.hypot(d2["UGRD200"], d2["VGRD200"])
    d2["shear"] = np.hypot(d2["UGRD200"] - d2["UGRD850"], d2["VGRD200"] - d2["VGRD850"])
    d2["thick"] = d2["HGT500"] - d2["HGT850"]
    d2["lapse"] = d2["TMP850"] - d2["TMP500"]
    d2["rh_mid"] = d2[["RH850", "RH750", "RH500"]].mean(axis=1)
    g = d2.groupby(["station", "date"])
    n = g["APCP"].count()
    mean_cols = ["T2M", "MSLP", "RH850", "RH750", "RH500", "rh_mid", "TMP850", "TMP500", "HGT850",
                 "HGT500", "UGRD850", "VGRD850", "UGRD200", "VGRD200", "wspd10", "wspd850",
                 "wspd200", "shear", "thick", "lapse"]
    out = g[mean_cols].mean()
    # APCP: only the 06Z/12Z/18Z stamps of d-1. If an APCP stamp means "accumulation over the
    # NEXT 6 h", the 00Z(d) stamp would cover 00-06Z, 3 h inside the next IMD day (leak).
    safe = d2[d2["time"].dt.hour != 0].groupby(["station", "date"])["APCP"]
    out["APCP"] = safe.sum()
    out["APCP_max6h"] = safe.max()
    out["MSLP_d6"] = g["MSLP"].last() - g["MSLP"].first()
    out = out[n == 4].add_prefix("i_").reset_index()
    out = out[(out["date"] >= "1990-01-01") & (out["date"] <= "2020-12-31")]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_PROCESSED / "d2_daily.parquet", index=False)
    print(out.shape, out["date"].min(), out["date"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
