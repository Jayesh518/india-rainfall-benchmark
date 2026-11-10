"""Stage 15: sample the IMDAA (1.08 deg) fields at the study stations.

Output: data/interim/d2_station_6h.parquet (time, station, ~20 columns), plus a
manifest entry for the source files actually used.
"""

from __future__ import annotations

import sys

from indiarain.config import DATA_INTERIM, STATIONS
from indiarain.data import d2_imdaa, provenance


def main() -> int:
    stations = tuple(s for s in STATIONS if s.name != "Port Blair")
    surf = d2_imdaa.surface_frame(stations)
    print("surface", surf.shape, flush=True)
    lev = d2_imdaa.level_frame(stations)
    print("levels", lev.shape, flush=True)
    df = surf.join(lev, how="left").reset_index()
    out = DATA_INTERIM / "d2_station_6h.parquet"
    df.to_parquet(out, index=False)
    print(df.shape, df["time"].min(), df["time"].max())
    print(df.isna().mean().round(4).to_string())

    used = [d2_imdaa.D2_DIR / "Surface_variables" / f for f in d2_imdaa.SURFACE.values()]
    for var in d2_imdaa.LEVELS:
        used += sorted((d2_imdaa.D2_DIR / "Atmospheric_variable").glob(f"IMDAA_{var}_prl_*/*.nc"))
    provenance.record(
        "D2",
        source="IndiaWeatherBench IMDAA copy (Kaggle, hand-downloaded by the owner); upstream "
        "NCMRWF IMDAA",
        licence="Upstream IMDAA is CC BY-NC-SA 4.0; no licence file ships with the copy "
        "(unverified). Attribute NCMRWF/MoES.",
        files=used,
        verified=False,
        note="1.08 deg (32x32) 6-hourly, 1990-2020 (T2m to 2023). Bilinear sampling at 34 "
        "stations. Not the 12 km IMDAA of the plan.",
        extra={"stations": len(stations), "interim": str(out.name)},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
