"""D1: IMD 0.25 deg gridded daily rainfall, the label for every experiment.

The grid is built from rain gauges over Indian land only, so some stations
(for example Port Blair in the Andamans) have no valid cell. Rather than
silently borrow a distant mainland cell, each station is matched to the nearest
cell *with data*, and stations further than ``MAX_MATCH_KM`` are dropped and
reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from indiarain.config import DATA_RAW, STATIONS, Station

D1_DIR: Path = DATA_RAW / "d1_imd"
SOURCE_URL = "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
#: One grid cell (~39 km diagonal at Indian latitudes) is the farthest we accept.
MAX_MATCH_KM = 40.0
#: A cell must have data on at least this share of days to be usable.
MIN_VALID_FRACTION = 0.95


def grid_file(year: int, root: Path = D1_DIR) -> Path:
    return root / "rain" / f"{year}.grd"


def download(start: int, end: int, root: Path = D1_DIR) -> list[Path]:
    """Download yearly GRD files with imdlib, skipping ones already present."""
    import imdlib as imd

    paths = []
    for year in range(start, end + 1):
        f = grid_file(year, root)
        if not (f.exists() and f.stat().st_size > 20_000_000):
            imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(root))
        paths.append(f)
    return paths


def haversine_km(
    lat1: float, lon1: float, lat2: npt.NDArray[np.float64], lon2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Great-circle distance from one point to arrays of points, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class CellMatch:
    station: str
    lat: float
    lon: float
    distance_km: float | None
    valid_fraction: float | None
    accepted: bool


def match_cells(
    valid_fraction: npt.ArrayLike,
    lats: npt.NDArray[np.float64],
    lons: npt.NDArray[np.float64],
    stations: tuple[Station, ...] = STATIONS,
) -> list[CellMatch]:
    """Nearest usable grid cell for each station.

    ``valid_fraction`` is a ``(lat, lon)`` array: share of days with data.
    """
    vf = np.asarray(valid_fraction)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    usable = vf >= MIN_VALID_FRACTION
    out: list[CellMatch] = []
    for s in stations:
        dist = haversine_km(s.lat, s.lon, lat_grid, lon_grid)
        dist = np.where(usable, dist, np.inf)
        i = np.unravel_index(np.argmin(dist), dist.shape)
        d = float(dist[i])
        if not math.isfinite(d):
            out.append(CellMatch(s.name, float("nan"), float("nan"), None, None, False))
            continue
        out.append(
            CellMatch(
                s.name, float(lats[i[0]]), float(lons[i[1]]), d, float(vf[i]), d <= MAX_MATCH_KM
            )
        )
    return out


def load_year_array(
    year: int, root: Path = D1_DIR
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.float64], pd.DatetimeIndex]:
    """Read one year as ``(time, lat, lon)`` float32 with NaN for missing."""
    import imdlib as imd

    ds = imd.open_data("rain", year, year, "yearwise", str(root)).get_xarray()
    arr = ds["rain"].values.astype("float32")
    arr[arr < 0] = np.nan  # defensive: IMD's missing sentinel is -999
    return arr, ds["lat"].values, ds["lon"].values, pd.DatetimeIndex(ds["time"].values)


def extract_stations(
    start: int,
    end: int,
    root: Path = D1_DIR,
    match_years: tuple[int, int] = (1979, 2024),
) -> tuple[pd.DataFrame, list[CellMatch]]:
    """Daily rainfall at each accepted station's matched cell, ``start``-``end``.

    Cells are chosen by data coverage over ``match_years`` (the supervised study
    window), so a cell with a gappy 1900s record is not rejected for climatology
    use. Returns a long frame ``[date, station, rain_mm]`` and the match report.
    """
    counts = None
    total = 0
    lats = lons = None
    for year in range(match_years[0], match_years[1] + 1):
        arr, lats, lons, times = load_year_array(year, root)
        valid = np.isfinite(arr).sum(axis=0)
        counts = valid if counts is None else counts + valid
        total += len(times)
    assert counts is not None and lats is not None and lons is not None
    matches = match_cells(counts / total, lats, lons)
    accepted = {m.station: m for m in matches if m.accepted}

    frames = []
    for year in range(start, end + 1):
        arr, lats, lons, times = load_year_array(year, root)
        for name, m in accepted.items():
            i = int(np.argmin(np.abs(lats - m.lat)))
            j = int(np.argmin(np.abs(lons - m.lon)))
            frames.append(pd.DataFrame({"date": times, "station": name, "rain_mm": arr[:, i, j]}))
    return pd.concat(frames, ignore_index=True), matches
