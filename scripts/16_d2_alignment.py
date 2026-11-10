"""Stage 16 (Phase 2): which 6-hourly IMDAA window matches the IMD rainfall day?

IMD day t = 24 h ending 08:30 IST = 03:00 UTC on day t. For shifts k (hours) the
IMDAA APCP is summed over stamps in (03Z(t-1)+k, 03Z(t)+k] and correlated with IMD(t)
(1990-2020, 34 stations). The best k tells how APCP stamps are labelled.
Output: reports/eda/d2_alignment.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import DATA_INTERIM, ROOT

OUT = ROOT / "reports" / "eda"


def main() -> int:
    d2 = pd.read_parquet(DATA_INTERIM / "d2_station_6h.parquet")
    imd = pd.read_parquet(DATA_INTERIM / "d1_station_rain.parquet")
    imd["date"] = pd.to_datetime(imd["date"])
    imd = imd.set_index(["station", "date"])["rain_mm"]
    rows = []
    for k in (-18, -12, -6, 0, 6, 12, 18):
        # stamp s belongs to the IMD day whose window (03Z(d-1)+k, 03Z(d)+k] contains it
        d = (d2["time"] - pd.Timedelta(hours=3 + k) - pd.Timedelta(seconds=1)).dt.floor("D") + pd.Timedelta(days=1)
        agg = d2.assign(date=d).groupby(["station", "date"])["APCP"].agg(["sum", "count"])
        agg = agg[agg["count"] == 4]["sum"]
        j = pd.concat([agg.rename("imdaa"), imd.rename("imd")], axis=1, join="inner").dropna()
        per = j.groupby(level=0).apply(lambda x: x["imdaa"].corr(x["imd"]))
        rows.append({"shift_h": k, "mean_corr": per.mean(), "median_corr": per.median(), "n": len(j)})
    res = pd.DataFrame(rows).round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "d2_alignment.csv", index=False)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
