"""Stage 20: hourly POWER -> daily features aligned to the IMD day (h_ prefix).

IMD day d covers the 24 hourly stamps 03Z(d-1) .. 02Z(d). POWER stamps are hour-beginning, so
the 03Z stamp (03-04 UTC = 08:30-09:30 IST) already lies inside the NEXT IMD day; including it
leaked (stage 22: a deliberate +1 h overlap adds 0.02 Brier skill, ending at 03Z adds 0.05). All of them close before the next IMD day opens, so no lag is needed. "Last 6 h" features
(stamps 21Z-02Z) give the recent state just before the target window opens.
Output: data/processed/power_hourly_daily.parquet
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import DATA_PROCESSED, DATA_RAW


def main() -> int:
    frames = []
    for f in sorted((DATA_RAW / "d3_power_hourly").glob("*.parquet")):
        station, _ = f.stem.rsplit("_", 1)
        df = pd.read_parquet(f).reset_index()
        df["station"] = station.replace("_", " ")
        frames.append(df)
    h = pd.concat(frames, ignore_index=True)
    shifted = h["time"] - pd.Timedelta(hours=3)
    h["date"] = shifted.dt.floor("D") + pd.Timedelta(days=1)
    h["last6"] = shifted.dt.hour >= 18  # stamps 21Z..02Z
    g = h.groupby(["station", "date"])
    n = g["T2M"].count()
    out = pd.DataFrame({
        "rain_sum": g["PRECTOTCORR"].sum(),
        "rain_max1h": g["PRECTOTCORR"].max(),
        "T2M_mean": g["T2M"].mean(), "T2M_max": g["T2M"].max(), "T2M_min": g["T2M"].min(),
        "RH_mean": g["RH2M"].mean(), "RH_max": g["RH2M"].max(),
        "QV_mean": g["QV2M"].mean(), "PS_mean": g["PS"].mean(),
        "PS_d": g["PS"].last() - g["PS"].first(),
        "WS_mean": g["WS10M"].mean(), "WS_max": g["WS10M"].max(),
        "SW_sum": g["ALLSKY_SFC_SW_DWN"].sum(),
    })
    last = h[h["last6"]].groupby(["station", "date"])
    out["rain_last6"] = last["PRECTOTCORR"].sum()
    out["RH_last6"] = last["RH2M"].mean()
    out["QV_last6"] = last["QV2M"].mean()
    out["PS_last6_d"] = last["PS"].last() - last["PS"].first()
    out["T_last6"] = last["T2M"].mean()
    out = out[n == 24].add_prefix("h_").reset_index()
    out.to_parquet(DATA_PROCESSED / "power_hourly_daily.parquet", index=False)
    print(out.shape, out["date"].min(), out["date"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
