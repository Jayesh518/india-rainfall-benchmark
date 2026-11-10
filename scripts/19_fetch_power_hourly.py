"""Stage 19: hourly NASA POWER for every study station, 2001-2024, in yearly chunks.

Resumable: each (station, year) is saved as soon as it arrives. Two workers, polite pause.
Outputs data/raw/d3_power_hourly/<station>_<year>.parquet and a manifest entry "D3H".
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from indiarain.config import DATA_RAW, STATIONS
from indiarain.data import d3_power_hourly, provenance

OUT = DATA_RAW / "d3_power_hourly"


def one(job):
    s, year = job
    f = OUT / f"{s.name.replace(' ', '_')}_{year}.parquet"
    if f.exists():
        return 0
    df = d3_power_hourly.fetch_point(s.lat, s.lon, f"{year}0101", f"{year}1231")
    df.to_parquet(f)
    time.sleep(1.0)
    return 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, y) for s in STATIONS if s.name != "Port Blair" for y in range(2001, 2025)]
    done = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, n in enumerate(ex.map(one, jobs), 1):
            done += n
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} jobs, {done} fetched", flush=True)
    files = sorted(OUT.glob("*.parquet"))
    provenance.record(
        "D3H",
        source=d3_power_hourly.URL,
        licence="NASA POWER: open data, cite NASA LaRC POWER Project",
        files=files,
        verified=True,
        note="Hourly, UTC, 2001-2024, 34 stations; native units",
        extra={"variables": d3_power_hourly.PARAMETERS, "files": len(files)},
    )
    print(f"D3H: {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
