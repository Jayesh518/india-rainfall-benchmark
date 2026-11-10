"""D5: climate-mode indices that modulate Indian monsoon rainfall.

* Nino 3.4 anomaly (ENSO), NOAA PSL, monthly
* SOI (ENSO), NOAA PSL, monthly
* DMI (Indian Ocean Dipole, HadISST), NOAA PSL, monthly
* OMI (OLR-based MJO index), NOAA PSL, daily

The plan named the BoM RMM index for the MJO, but bom.gov.au blocks automated
downloads, so the NOAA PSL OMI index is used instead. This is a recorded
deviation, not a silent swap.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from indiarain.data import http

PSL = "https://psl.noaa.gov"
MONTHLY_SOURCES: dict[str, str] = {
    "nino34": f"{PSL}/data/correlation/nina34.anom.data",
    "soi": f"{PSL}/data/correlation/soi.data",
    "dmi": f"{PSL}/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
}
OMI_URL = f"{PSL}/mjo/mjoindex/omi.1x.txt"
#: PSL marks missing monthly values with large negative numbers.
MISSING_THRESHOLD = -90.0


def parse_psl_monthly(text: str, name: str) -> pd.Series:
    """Parse PSL's ``YEAR v1..v12`` table into a month-start series.

    The first line (year range) and trailing footer lines are skipped; missing
    values become NaN.
    """
    rows: dict[pd.Timestamp, float] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        for month, v in enumerate(vals, start=1):
            rows[pd.Timestamp(year=year, month=month, day=1)] = (
                np.nan if v <= MISSING_THRESHOLD else v
            )
    s = pd.Series(rows, name=name, dtype=float).sort_index()
    s.index.name = "month"
    return s


def parse_omi(text: str) -> pd.DataFrame:
    """OMI daily file: ``year month day PC1 PC2 amplitude``; fill value 1e36."""
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        header=None,
        names=["year", "month", "day", "omi_pc1", "omi_pc2", "omi_amp"],
        comment="#",
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date").drop(columns=["year", "month", "day"])
    return df.where(df.abs() < 1e10)


def fetch_monthly() -> pd.DataFrame:
    series = [
        parse_psl_monthly(http.get(url).text, name) for name, url in MONTHLY_SOURCES.items()
    ]
    return pd.concat(series, axis=1)


def fetch_omi() -> pd.DataFrame:
    return parse_omi(http.get(OMI_URL).text)
