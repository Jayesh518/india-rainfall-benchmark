"""D4h: ERA5 hourly through Open-Meteo's archive API, UTC time.

Needs hourly stamps to cut the day at the IMD boundary. Open-Meteo's precipitation is the sum of
the PRECEDING hour, instantaneous variables are valid at the stamp. At most 10 variables per
request keeps the quota weight at max(1, days/14), the same as the daily product.
"""

from __future__ import annotations

import pandas as pd

from indiarain.data import http
from indiarain.data.d4_openmeteo import URL, QuotaExceededError

HOURLY: dict[str, str] = {
    "temperature_2m": "degC",
    "relative_humidity_2m": "%",
    "dew_point_2m": "degC",
    "surface_pressure": "hPa",
    "pressure_msl": "hPa",
    "cloud_cover": "%",
    "wind_speed_10m": "km/h",
    "precipitation": "mm",
    "shortwave_radiation": "W/m^2",
}


def fetch_point(lat: float, lon: float, start: str, end: str) -> tuple[pd.DataFrame, tuple[float, float]]:
    resp = http.get_once(
        URL,
        params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "models": "era5", "timezone": "UTC", "hourly": ",".join(HOURLY)},
        timeout=300,
    )
    if resp.status_code == 429:
        try:
            reason = str(resp.json().get("reason", "rate limit"))
        except ValueError:
            reason = "rate limit"
        raise QuotaExceededError(reason)
    resp.raise_for_status()
    p = resp.json()
    h = p["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"})
    df.index = pd.to_datetime(h["time"])
    df.index.name = "time"
    return df, (p["latitude"], p["longitude"])
