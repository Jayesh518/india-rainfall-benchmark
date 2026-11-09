"""Project-wide constants: paths, IMD conventions, and the station set.

Centralised so that no threshold or city coordinate is ever duplicated in a
notebook. Everything here follows India Meteorological Department convention
rather than the Australian conventions common in the rainfall-ML literature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_RAW: Final[Path] = ROOT / "data" / "raw"
DATA_INTERIM: Final[Path] = ROOT / "data" / "interim"
DATA_PROCESSED: Final[Path] = ROOT / "data" / "processed"
DATA_EXTERNAL: Final[Path] = ROOT / "data" / "external"
RESULTS: Final[Path] = ROOT / "results"
MODELS: Final[Path] = ROOT / "models"

# --------------------------------------------------------------------------- #
# IMD rainfall conventions
# --------------------------------------------------------------------------- #

#: A day is a "rainy day" at >= 2.5 mm (IMD). The Australian weatherAUS
#: convention is 1.0 mm; we report both so results stay comparable, but 2.5 mm
#: is the primary threshold.
RAINY_DAY_MM: Final[float] = 2.5
RAINY_DAY_MM_ALT: Final[float] = 1.0

#: Official IMD daily rainfall intensity categories, as (label, lower, upper)
#: in mm with the lower bound inclusive. Used for the ordinal task, which gives
#: a per-model probability vector rather than a single yes/no.
RAINFALL_CATEGORIES: Final[tuple[tuple[str, float, float], ...]] = (
    ("No rain", 0.0, 2.5),
    ("Light", 2.5, 15.6),
    ("Moderate", 15.6, 64.5),
    ("Heavy", 64.5, 115.6),
    ("Very heavy", 115.6, 204.5),
    ("Extremely heavy", 204.5, float("inf")),
)

#: Threshold for the extreme-event task (~1-2% positives).
HEAVY_RAIN_MM: Final[float] = 64.5

#: Indian monsoon seasons (IMD): inclusive month ranges.
SEASONS: Final[dict[str, tuple[int, int]]] = {
    "winter": (1, 2),
    "pre_monsoon": (3, 5),
    "southwest_monsoon": (6, 9),
    "post_monsoon": (10, 12),
}

# --------------------------------------------------------------------------- #
# Station set
# --------------------------------------------------------------------------- #


class Station(NamedTuple):
    """A study location. ``zone`` is a coarse climate grouping used for the
    leave-one-zone-out generalisation test."""

    name: str
    lat: float
    lon: float
    state: str
    zone: str


#: 35 cities chosen to span India's climate zones rather than just the metros,
#: so that leave-one-zone-out tests are meaningful.
STATIONS: Final[tuple[Station, ...]] = (
    Station("Mumbai", 19.08, 72.88, "Maharashtra", "west_coast"),
    Station("Pune", 18.52, 73.86, "Maharashtra", "interior_peninsula"),
    Station("Nagpur", 21.15, 79.09, "Maharashtra", "central"),
    Station("Surat", 21.17, 72.83, "Gujarat", "west_coast"),
    Station("Ahmedabad", 23.03, 72.59, "Gujarat", "semi_arid"),
    Station("Jodhpur", 26.24, 73.02, "Rajasthan", "arid"),
    Station("Jaipur", 26.91, 75.79, "Rajasthan", "semi_arid"),
    Station("Delhi", 28.61, 77.21, "Delhi", "northern_plains"),
    Station("Chandigarh", 30.73, 76.78, "Chandigarh", "northern_plains"),
    Station("Shimla", 31.10, 77.17, "Himachal Pradesh", "western_himalaya"),
    Station("Srinagar", 34.08, 74.80, "Jammu & Kashmir", "western_himalaya"),
    Station("Leh", 34.16, 77.58, "Ladakh", "cold_desert"),
    Station("Dehradun", 30.32, 78.03, "Uttarakhand", "western_himalaya"),
    Station("Lucknow", 26.85, 80.95, "Uttar Pradesh", "northern_plains"),
    Station("Patna", 25.59, 85.14, "Bihar", "northern_plains"),
    Station("Ranchi", 23.34, 85.31, "Jharkhand", "eastern_plateau"),
    Station("Kolkata", 22.57, 88.36, "West Bengal", "gangetic_delta"),
    Station("Bhubaneswar", 20.30, 85.82, "Odisha", "east_coast"),
    Station("Raipur", 21.25, 81.63, "Chhattisgarh", "central"),
    Station("Bhopal", 23.26, 77.41, "Madhya Pradesh", "central"),
    Station("Indore", 22.72, 75.86, "Madhya Pradesh", "central"),
    Station("Guwahati", 26.14, 91.74, "Assam", "northeast"),
    Station("Shillong", 25.58, 91.89, "Meghalaya", "northeast"),
    Station("Agartala", 23.83, 91.28, "Tripura", "northeast"),
    Station("Imphal", 24.82, 93.94, "Manipur", "northeast"),
    Station("Gangtok", 27.33, 88.61, "Sikkim", "eastern_himalaya"),
    Station("Visakhapatnam", 17.69, 83.22, "Andhra Pradesh", "east_coast"),
    Station("Hyderabad", 17.39, 78.49, "Telangana", "interior_peninsula"),
    Station("Bengaluru", 12.97, 77.59, "Karnataka", "interior_peninsula"),
    Station("Chennai", 13.08, 80.27, "Tamil Nadu", "east_coast"),
    Station("Madurai", 9.93, 78.12, "Tamil Nadu", "interior_peninsula"),
    Station("Coimbatore", 11.02, 76.96, "Tamil Nadu", "interior_peninsula"),
    Station("Kochi", 9.93, 76.27, "Kerala", "west_coast"),
    Station("Thiruvananthapuram", 8.52, 76.94, "Kerala", "west_coast"),
    Station("Port Blair", 11.62, 92.73, "Andaman & Nicobar", "islands"),
)

#: Study period. D1 (IMD rainfall) reaches back to 1901, but D2 (IMDAA
#: predictors) starts in 1979, so the supervised benchmark starts there.
#: Climatology and extreme-value work may use the full D1 record.
STUDY_START_YEAR: Final[int] = 1979
STUDY_END_YEAR: Final[int] = 2024
CLIMATOLOGY_START_YEAR: Final[int] = 1901


def categorise_rainfall(mm: float) -> str:
    """Map a daily rainfall total (mm) to its IMD intensity category.

    >>> categorise_rainfall(0.0)
    'No rain'
    >>> categorise_rainfall(2.5)
    'Light'
    >>> categorise_rainfall(80.0)
    'Heavy'
    """
    if mm < 0:
        raise ValueError(f"rainfall cannot be negative: {mm}")
    for label, low, high in RAINFALL_CATEGORIES:
        if low <= mm < high:
            return label
    return RAINFALL_CATEGORIES[-1][0]


def station_by_name(name: str) -> Station:
    """Look up a station, case-insensitively."""
    key = name.strip().casefold()
    for s in STATIONS:
        if s.name.casefold() == key:
            return s
    raise KeyError(f"unknown station: {name!r}")


def zones() -> tuple[str, ...]:
    """Distinct climate zones, for leave-one-zone-out splits."""
    return tuple(dict.fromkeys(s.zone for s in STATIONS))
