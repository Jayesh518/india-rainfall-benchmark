"""Stage 26: hourly ERA5 for every station from the Copernicus CDS (no Open-Meteo quota).

Four quarterly requests per station-year (CDS cost limit rejects 6 months of 9 variables), a 0.5 deg box
around the station, nearest ERA5 cell kept. Output columns and units match stage 23 so stage 24
can read either source: temperature_2m degC, relative_humidity_2m %, dew_point_2m degC,
surface_pressure hPa, pressure_msl hPa, cloud_cover %, wind_speed_10m km/h, precipitation mm
(sum of the PRECEDING hour, ERA5 accumulation), shortwave_radiation W/m2 (hourly mean).
Resumable: finished station-years are skipped. Output: data/raw/d4_cds/<station>_<year>.parquet
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from indiarain.config import DATA_RAW, STATIONS
from indiarain.data import provenance

OUT = DATA_RAW / "d4_cds"
VARS = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure", "mean_sea_level_pressure",
        "total_cloud_cover", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "total_precipitation", "surface_solar_radiation_downwards"]


def convert(ds: xr.Dataset, lat: float, lon: float) -> tuple[pd.DataFrame, tuple[float, float]]:
    if "expver" in ds.dims:  # ERA5T/ERA5 mix: take whichever is present
        ds = ds.reduce(np.nanmax, dim="expver")
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
    t = ds["valid_time"].to_numpy() if "valid_time" in ds else ds["time"].to_numpy()
    k = pd.DataFrame({n: ds[n].to_numpy() for n in ds.data_vars}, index=pd.DatetimeIndex(t, name="time"))
    tc, td = k["t2m"] - 273.15, k["d2m"] - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * tc / (243.04 + tc))
    out = pd.DataFrame({
        "temperature_2m": tc, "relative_humidity_2m": rh.clip(0, 100), "dew_point_2m": td,
        "surface_pressure": k["sp"] / 100, "pressure_msl": k["msl"] / 100,
        "cloud_cover": k["tcc"] * 100, "wind_speed_10m": np.hypot(k["u10"], k["v10"]) * 3.6,
        "precipitation": (k["tp"] * 1000).clip(lower=0), "shortwave_radiation": k["ssrd"] / 3600,
    })
    return out, (float(ds["latitude"]), float(ds["longitude"]))


def fetch(job: tuple) -> str:
    s, year = job
    final = OUT / f"{s.name.replace(' ', '_')}_{year}.parquet"
    if final.exists():
        return f"{s.name} {year} cached"
    area = [s.lat + 0.25, s.lon - 0.25, s.lat - 0.25, s.lon + 0.25]
    parts, cell = [], None
    for months in (range(1, 4), range(4, 7), range(7, 10), range(10, 13)):  # CDS cost limit: 3 months ok, 6 rejected
        with tempfile.TemporaryDirectory() as tmp:
            nc = Path(tmp) / "r.nc"
            cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
                "product_type": ["reanalysis"], "variable": VARS, "year": [str(year)],
                "month": [f"{m:02d}" for m in months], "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)], "area": area,
                "data_format": "netcdf", "download_format": "unarchived"}, str(nc))
            with xr.open_dataset(nc) as ds:
                part, cell = convert(ds.load(), s.lat, s.lon)
        parts.append(part)
    df = pd.concat(parts)
    df.to_parquet(final)
    return f"{s.name} {year} ok cell {cell} rows {len(df)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--station", action="append")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stations = [s for s in STATIONS if s.name != "Port Blair"
                and (not args.station or s.name in args.station)]
    jobs = [(s, y) for y in range(args.start, args.end + 1) for s in stations]
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for _job, res in zip(jobs, ex.map(lambda j: _safe(j), jobs), strict=True):
            failed += res.startswith("FAIL")
            print(res, flush=True)
    files = sorted(OUT.glob("*.parquet"))
    provenance.record(
        "D4C", source="https://cds.climate.copernicus.eu (reanalysis-era5-single-levels)",
        licence="CC-BY; 'Generated using Copernicus Climate Change Service information' "
        "(Hersbach et al. 2023, doi:10.24381/cds.adbb2d47)",
        files=files, verified=failed == 0 and len(files) == len(stations) * (args.end - args.start + 1),
        note="Hourly ERA5, 0.25 deg, UTC stamps, nearest cell to each station.",
        extra={"variables": VARS, "failed": failed})
    print(f"{len(files)} station-years on disk, {failed} failed")
    return 1 if failed else 0


def _safe(job: tuple) -> str:
    """Retry while CDS says too many of our requests are queued (orphans drain in time)."""
    for _ in range(200):
        try:
            return fetch(job)
        except Exception as exc:
            msg = str(exc)
            if "queued requests" in msg or "has been rejected" in msg:
                time.sleep(300)
                continue
            return f"FAIL {job[0].name} {job[1]}: {msg[:200]}"
    return f"FAIL {job[0].name} {job[1]}: queue never drained"


if __name__ == "__main__":
    sys.exit(main())
