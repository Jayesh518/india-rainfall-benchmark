"""Stage 04: fetch D4 (ERA5 via Open-Meteo) daily data for every study station.

Open-Meteo's free tier allows 10,000 weighted calls a day, and one ten-year
request costs about 260, so the full fetch (34 stations x 44 years) needs
several days. This stage is therefore resumable at the chunk level: every
ten-year chunk is saved as soon as it arrives, and when Open-Meteo reports a
rate limit the script stops cleanly (exit code 3) instead of burning retries.
Re-run it later and it continues where it stopped.

Outputs
    data/raw/d4_era5/_chunks/<station>_<start>.parquet   finished chunks
    data/raw/d4_era5/<station>.parquet                   assembled stations
    data/manifest.json                                   provenance entry "D4"
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from indiarain.config import DATA_RAW, STATIONS
from indiarain.data import d4_openmeteo, provenance

D4_DIR = DATA_RAW / "d4_era5"
CHUNK_DIR = D4_DIR / "_chunks"
QUOTA_EXIT = 3


def chunks(start: int, end: int, size: int = 10) -> list[tuple[int, str, str]]:
    """``(first_year, start_date, end_date)`` per chunk."""
    return [
        (y, f"{y}-01-01", f"{min(y + size - 1, end)}-12-31") for y in range(start, end + 1, size)
    ]


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=1981, help="matches D3, which begins 1981")
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--pause", type=float, default=15.0, help="seconds between requests")
    ap.add_argument("--station", action="append", help="limit to these stations")
    args = ap.parse_args(argv)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [
        s for s in STATIONS if s.name != "Port Blair" and (not args.station or s.name in args.station)
    ]
    plan = chunks(args.start, args.end)
    cell_file = CHUNK_DIR / "_cells.json"
    cells: dict[str, list[float]] = json.loads(cell_file.read_text()) if cell_file.exists() else {}

    stopped = None
    for s in wanted:
        slug = _slug(s.name)
        final = D4_DIR / f"{slug}.parquet"
        if final.exists():
            continue
        for first, a, b in plan:
            part_file = CHUNK_DIR / f"{slug}_{first}.parquet"
            if part_file.exists():
                continue
            try:
                try:
                    part, cell = d4_openmeteo.fetch_point(s.lat, s.lon, a, b)
                except d4_openmeteo.QuotaExceededError as exc:
                    if "minutely" not in exc.reason.lower():
                        raise
                    print("  minute limit hit; waiting 65 s", flush=True)
                    time.sleep(65)
                    part, cell = d4_openmeteo.fetch_point(s.lat, s.lon, a, b)
            except d4_openmeteo.QuotaExceededError as exc:
                stopped = exc.reason
                break
            part.to_parquet(part_file)
            cells[s.name] = list(cell)
            cell_file.write_text(json.dumps(cells, indent=1))
            print(f"  {s.name:<20} {a[:4]}-{b[:4]}  ok", flush=True)
            time.sleep(args.pause)
        if stopped:
            break
        parts = [pd.read_parquet(CHUNK_DIR / f"{slug}_{first}.parquet") for first, _, _ in plan]
        df = pd.concat(parts)
        df.to_parquet(final)
        print(f"{s.name:<20} {len(df):>6} days  ERA5 cell {cells.get(s.name)}  "
              f"NaN {df.isna().mean().mean():.1%}", flush=True)

    files = sorted(D4_DIR.glob("*.parquet"))
    done = {f.stem.replace("_", " ") for f in files}
    pending = [s.name for s in wanted if s.name not in done]
    provenance.record(
        "D4",
        source=d4_openmeteo.URL,
        licence="Free for non-commercial use; cite Zippenfenig (2023) DOI 10.5281/ZENODO.7970649 "
        "and 'Generated using Copernicus Climate Change Service information'",
        files=files,
        verified=not pending,
        note="ERA5 (models=era5), 0.25 deg, daily aggregates in Asia/Kolkata local time. "
        + ("COMPLETE." if not pending else f"INCOMPLETE: {len(pending)} stations pending (quota)."),
        extra={
            "variables": d4_openmeteo.DAILY,
            "start": args.start,
            "end": args.end,
            "stations": len(files),
            "pending": pending,
            "era5_cells": cells,
        },
    )
    if stopped:
        print(f"\nStopped by Open-Meteo: {stopped}")
        print(f"D4: {len(files)} stations done, {len(pending)} pending. Re-run later to resume.")
        return QUOTA_EXIT
    print(f"\nD4: {len(files)} station files, pending: {pending or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
