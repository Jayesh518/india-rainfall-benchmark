"""D4: ERA5 reanalysis through Open-Meteo's archive API.

Attribution required: Zippenfenig, P. (2023) Open-Meteo.com Weather API,
DOI 10.5281/ZENODO.7970649; "Generated using Copernicus Climate Change Service
information". Days are aggregated in Asia/Kolkata local time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from indiarain.data import http

URL = "https://archive-api.open-meteo.com/v1/archive"

#: Free tier: 600 weighted calls/min, 5,000/hour, 10,000/day. A request costs
#: roughly max(variables/10, days/14), so ten years of daily data is ~260 calls
#: and a full 44-year, 34-station fetch needs ~40,000, i.e. several days.
DAILY_QUOTA = 10_000


class QuotaExceededError(RuntimeError):
    """Open-Meteo refused the request because a rate limit was reached."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def request_weight(n_days: int, n_variables: int | None = None) -> float:
    """Approximate quota cost of one request."""
    n_vars = len(DAILY) if n_variables is None else n_variables
    return max(n_vars / 10, n_days / 14)


DAILY: dict[str, str] = {
    "temperature_2m_max": "degC",
    "temperature_2m_min": "degC",
    "temperature_2m_mean": "degC",
    "relative_humidity_2m_mean": "%",
    "dew_point_2m_mean": "degC",
    "surface_pressure_mean": "hPa",
    "pressure_msl_mean": "hPa",
    "cloud_cover_mean": "%",
    "wind_speed_10m_max": "km/h",
    "wind_direction_10m_dominant": "degrees",
    "shortwave_radiation_sum": "MJ/m^2",
    "et0_fao_evapotranspiration": "mm",
    "precipitation_sum": "mm",
}


def parse(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload["daily"]
    df = pd.DataFrame({k: v for k, v in daily.items() if k != "time"})
    df.index = pd.to_datetime(daily["time"])
    df.index.name = "date"
    return df


def fetch_point(
    lat: float, lon: float, start: str, end: str
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """``start``/``end`` as YYYY-MM-DD. Returns the frame and the ERA5 cell used."""
    resp = http.get_once(
        URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "models": "era5",
            "timezone": "Asia/Kolkata",
            "daily": ",".join(DAILY),
        },
        timeout=300,
    )
    if resp.status_code == 429:
        try:
            reason = str(resp.json().get("reason", "rate limit"))
        except ValueError:
            reason = "rate limit"
        raise QuotaExceededError(reason)
    resp.raise_for_status()
    payload = resp.json()
    return parse(payload), (payload["latitude"], payload["longitude"])
