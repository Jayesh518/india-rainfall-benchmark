"""D2: IMDAA reanalysis, IndiaWeatherBench copy (1.08 deg, 6-hourly, 1990-2020).

The files are hand-downloaded (Kaggle, owner's login). They are NOT the 12 km
IMDAA of the research plan: the grid is 32 x 32 over 5-38.5N, 65-98.5E, so one
cell is about 120 km. Station series are bilinearly interpolated to the
station coordinates. Licence is not stated in the folder (IMDAA is CC BY-NC-SA
4.0 upstream); recorded as unverified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from indiarain.config import DATA_RAW, Station

D2_DIR = DATA_RAW / "d2_imdaa"
SURFACE = {
    "APCP": "IMDAA_APCP_sfc_1.08_1990_2020.nc",
    "T2M": "IMDAA_TMP_2m_1.08_1990_2020.nc",
    "MSLP": "IMDAA_PRMSL_msl_1.08_1990_2020.nc",
    "U10": "IMDAA_UGRD_10m_1.08_1990_2020.nc",
    "V10": "IMDAA_VGRD_10m_1.08_1990_2020.nc",
}
#: (variable, hPa levels) taken from the yearly pressure-level files
LEVELS = {"RH": (850, 750, 500), "TMP": (850, 500), "HGT": (850, 500),
          "UGRD": (850, 200), "VGRD": (850, 200)}
START, END = "1990-01-01", "2020-12-31"


def _points(stations: tuple[Station, ...]) -> tuple[xr.DataArray, xr.DataArray]:
    names = [s.name for s in stations]
    lat = xr.DataArray([s.lat for s in stations], dims="station", coords={"station": names})
    lon = xr.DataArray([s.lon for s in stations], dims="station", coords={"station": names})
    return lat, lon


def surface_frame(stations: tuple[Station, ...], root: Path = D2_DIR) -> pd.DataFrame:
    lat, lon = _points(stations)
    cols = {}
    for name, fname in SURFACE.items():
        ds = xr.open_dataset(root / "Surface_variables" / fname)
        da = ds[next(iter(ds.data_vars))].sel(time=slice(START, END + " 23:59"))
        out = da.interp(latitude=lat, longitude=lon, method="linear")
        cols[name] = out.to_pandas().stack()
    return pd.DataFrame(cols).rename_axis(["time", "station"])


def level_frame(stations: tuple[Station, ...], root: Path = D2_DIR) -> pd.DataFrame:
    lat, lon = _points(stations)
    frames = []
    for var, levels in LEVELS.items():
        folder = root / "Atmospheric_variable" / f"IMDAA_{var}_prl_1.08_1990_2020"
        parts = []
        for f in sorted(folder.glob("*.nc")):
            ds = xr.open_dataset(f)
            da = ds[next(iter(ds.data_vars))].sel(plevel=list(levels))
            parts.append(da.interp(latitude=lat, longitude=lon, method="linear").load())
        full = xr.concat(parts, dim="time")
        for lev in levels:
            s = full.sel(plevel=lev, drop=True).to_pandas().stack()
            s.name = f"{var}{lev}"
            frames.append(s)
    return pd.concat(frames, axis=1).rename_axis(["time", "station"])
