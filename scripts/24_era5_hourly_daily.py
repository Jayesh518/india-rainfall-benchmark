"""Stage 24: hourly ERA5 (Open-Meteo, UTC) -> daily features aligned to the IMD day (w_ prefix).

Same window as stage 20: IMD day d uses stamps 03Z(d-1)..02Z(d). ERA5 precipitation is the sum of
the PRECEDING hour, so the 03Z(d) stamp would still be inside the window; ending at 02Z is one hour
more conservative than needed and keeps the POWER and ERA5 windows identical.
Only stations whose five-year chunks are all present are used.
Output: data/processed/era5_hourly_daily.parquet
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import DATA_PROCESSED, DATA_RAW

YEARS = (2001, 2006, 2011, 2016, 2021)


def main() -> int:
    frames, skipped = [], []
    base = DATA_RAW / "d4_era5_hourly"
    for slug in sorted({f.stem.rsplit("_", 1)[0] for f in base.glob("*.parquet")}):
        parts = [base / f"{slug}_{y}.parquet" for y in YEARS]
        if not all(p.exists() for p in parts):
            skipped.append(slug)
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts]).reset_index()
        df["station"] = slug.replace("_", " ")
        frames.append(df)
    h = pd.concat(frames, ignore_index=True)
    shifted = h["time"] - pd.Timedelta(hours=3)
    h["date"] = shifted.dt.floor("D") + pd.Timedelta(days=1)
    h["last6"] = shifted.dt.hour >= 18
    g = h.groupby(["station", "date"])
    n = g["temperature_2m"].count()
    out = pd.DataFrame({
        "rain_sum": g["precipitation"].sum(), "rain_max1h": g["precipitation"].max(),
        "T_mean": g["temperature_2m"].mean(), "T_max": g["temperature_2m"].max(),
        "T_min": g["temperature_2m"].min(),
        "RH_mean": g["relative_humidity_2m"].mean(), "RH_max": g["relative_humidity_2m"].max(),
        "Td_mean": g["dew_point_2m"].mean(), "PS_mean": g["surface_pressure"].mean(),
        "MSL_mean": g["pressure_msl"].mean(),
        "PS_d": g["surface_pressure"].last() - g["surface_pressure"].first(),
        "cloud_mean": g["cloud_cover"].mean(),
        "WS_mean": g["wind_speed_10m"].mean(), "WS_max": g["wind_speed_10m"].max(),
        "SW_sum": g["shortwave_radiation"].sum(),
    })
    last = h[h["last6"]].groupby(["station", "date"])
    out["rain_last6"] = last["precipitation"].sum()
    out["RH_last6"] = last["relative_humidity_2m"].mean()
    out["Td_last6"] = last["dew_point_2m"].mean()
    out["PS_last6_d"] = last["surface_pressure"].last() - last["surface_pressure"].first()
    out["cloud_last6"] = last["cloud_cover"].mean()
    out = out[n == 24].add_prefix("w_").reset_index()
    out.to_parquet(DATA_PROCESSED / "era5_hourly_daily.parquet", index=False)
    print(out.shape, out["station"].nunique(), "stations; skipped incomplete:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
