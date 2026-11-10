"""D3h: NASA POWER hourly point data (MERRA-2), UTC time standard.

Hourly values let the day be cut at the IMD boundary (03:00 UTC = 08:30 IST), which the
daily product cannot do. Available from 2001. Attribution as for the daily product.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indiarain.data import http

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS: dict[str, str] = {
    "T2M": "degC",
    "RH2M": "%",
    "QV2M": "g/kg",
    "PS": "kPa",
    "WS10M": "m/s",
    "PRECTOTCORR": "mm/hr",
    "ALLSKY_SFC_SW_DWN": "W/m^2",
}
FILL_VALUE = -999.0


def parse(payload: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(payload["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    df.index.name = "time"
    return df.where(df != FILL_VALUE)


def fetch_point(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """``start``/``end`` as YYYYMMDD (at most about a year per request)."""
    resp = http.get(
        URL,
        params={
            "parameters": ",".join(PARAMETERS),
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "UTC",
        },
        timeout=300,
    )
    return parse(resp.json())
