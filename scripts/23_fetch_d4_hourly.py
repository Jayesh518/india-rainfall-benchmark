"""Stage 23: hourly ERA5 (Open-Meteo) for the chosen stations, 2001-2024, resumable.

Five-year chunks saved on arrival; stops with exit code 3 on a quota limit (a per-minute limit
is waited out once). Output data/raw/d4_era5_hourly/<station>_<first_year>.parquet, manifest "D4H".
"""

from __future__ import annotations

import argparse
import sys
import time

from indiarain.config import DATA_RAW, STATIONS
from indiarain.data import d4_openmeteo_hourly as hm
from indiarain.data import provenance
from indiarain.data.d4_openmeteo import QuotaExceededError

OUT = DATA_RAW / "d4_era5_hourly"
QUOTA_EXIT = 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2001)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--pause", type=float, default=20.0)
    ap.add_argument("--station", action="append", required=True)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = [(y, f"{y}-01-01", f"{min(y + 4, args.end)}-12-31") for y in range(args.start, args.end + 1, 5)]
    stopped = None
    for s in [x for x in STATIONS if x.name in args.station]:
        for first, a, b in chunks:
            f = OUT / f"{s.name.replace(' ', '_')}_{first}.parquet"
            if f.exists():
                continue
            try:
                try:
                    df, cell = hm.fetch_point(s.lat, s.lon, a, b)
                except QuotaExceededError as exc:
                    if "minutely" not in exc.reason.lower():
                        raise
                    time.sleep(65)
                    df, cell = hm.fetch_point(s.lat, s.lon, a, b)
            except QuotaExceededError as exc:
                stopped = exc.reason
                break
            df.to_parquet(f)
            print(f"  {s.name:<14} {a[:4]}-{b[:4]} ok cell {cell}", flush=True)
            time.sleep(args.pause)
        if stopped:
            break
    files = sorted(OUT.glob("*.parquet"))
    provenance.record(
        "D4H", source=hm.URL if hasattr(hm, "URL") else "https://archive-api.open-meteo.com/v1/archive",
        licence="Free for non-commercial use; cite Zippenfenig (2023) DOI 10.5281/ZENODO.7970649 "
        "and 'Generated using Copernicus Climate Change Service information'",
        files=files, verified=False,
        note="ERA5 hourly, UTC, 5-year chunks. " + (f"STOPPED: {stopped}" if stopped else "complete for the requested stations"),
        extra={"variables": hm.HOURLY, "stations_requested": args.station},
    )
    if stopped:
        print("Stopped by Open-Meteo:", stopped)
        return QUOTA_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
