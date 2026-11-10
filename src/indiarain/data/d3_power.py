"""D3: NASA POWER daily point data (MERRA-2 derived), near-real-time predictors."""

from __future__ import annotations

from typing import Any

import pandas as pd

from indiarain.data import http

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
#: Parameter -> unit as reported by the API. Note PS is in kPa, not hPa.
PARAMETERS: dict[str, str] = {
    "T2M": "degC",
    "T2M_MAX": "degC",
    "T2M_MIN": "degC",
    "RH2M": "%",
    "QV2M": "g/kg",
    "PS": "kPa",
    "WS10M": "m/s",
    "WD10M": "degrees",
    "ALLSKY_SFC_SW_DWN": "kWh/m^2/day",
    "PRECTOTCORR": "mm/day",
}
FILL_VALUE = -999.0


def parse(payload: dict[str, Any]) -> pd.DataFrame:
    """POWER JSON -> frame indexed by date, fill values turned into NaN."""
    df = pd.DataFrame(payload["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    return df.where(df != FILL_VALUE)


def fetch_point(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """``start``/``end`` as YYYYMMDD. One request per station, whole range."""
    resp = http.get(
        URL,
        params={
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON",
        },
        timeout=300,
    )
    return parse(resp.json())
