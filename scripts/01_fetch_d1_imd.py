"""Stage 01: fetch D1 (IMD 0.25 deg gridded rainfall) and extract station series.

Downloads yearly grids (default 1901-2024), matches every study station to the
nearest grid cell that has data, and writes the long station table.

Outputs
    data/raw/d1_imd/rain/<year>.grd          raw yearly grids (git-ignored)
    data/interim/d1_station_rain.parquet     [date, station, rain_mm]
    data/manifest.json                       provenance entry "D1"
"""

from __future__ import annotations

import argparse
import sys

from indiarain.config import DATA_INTERIM
from indiarain.data import d1_imd, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=1901)
    ap.add_argument("--end", type=int, default=2024)
    args = ap.parse_args(argv)

    files = d1_imd.download(args.start, args.end)
    df, matches = d1_imd.extract_stations(args.start, args.end)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out = DATA_INTERIM / "d1_station_rain.parquet"
    df.to_parquet(out, index=False)

    dropped = [m.station for m in matches if not m.accepted]
    for m in matches:
        status = "ok " if m.accepted else "DROP"
        dist = "n/a" if m.distance_km is None else f"{m.distance_km:5.1f} km"
        print(f"[{status}] {m.station:<20} nearest data cell {dist}")

    provenance.record(
        "D1",
        source=d1_imd.SOURCE_URL,
        licence="IMD terms of use; cite India Meteorological Department, Pune",
        files=files,
        note=(
            "Downloaded with imdlib. Stations matched to the nearest grid cell with "
            f">={d1_imd.MIN_VALID_FRACTION:.0%} valid days over 1979-2024 and within "
            f"{d1_imd.MAX_MATCH_KM:.0f} km; others dropped."
        ),
        extra={
            "years": [args.start, args.end],
            "station_rows": len(df),
            "stations_kept": sorted({m.station for m in matches if m.accepted}),
            "stations_dropped": dropped,
            "cell_matches": [
                {
                    "station": m.station,
                    "cell_lat": m.lat,
                    "cell_lon": m.lon,
                    "distance_km": None if m.distance_km is None else round(m.distance_km, 1),
                    "accepted": m.accepted,
                }
                for m in matches
            ],
        },
    )
    print(f"\nD1: {len(files)} yearly grids, {len(df):,} station-day rows, dropped: {dropped or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
