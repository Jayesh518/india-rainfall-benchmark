"""Stage 03: fetch D3 (NASA POWER daily point data) for every study station.

One request per station for the whole range, with a short pause between
requests (POWER blocks clients that hammer it). Existing files are kept, so the
script can be re-run to resume.

Outputs
    data/raw/d3_power/<station>.parquet
    data/manifest.json   provenance entry "D3"
"""

from __future__ import annotations

import argparse
import sys
import time

from indiarain.config import DATA_RAW, STATIONS
from indiarain.data import d3_power, provenance

D3_DIR = DATA_RAW / "d3_power"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="19810101", help="POWER begins 1981-01-01")
    ap.add_argument("--end", default="20241231")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--station", action="append", help="limit to these stations")
    args = ap.parse_args(argv)

    D3_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in STATIONS if not args.station or s.name in args.station]
    failed = []
    for s in wanted:
        out = D3_DIR / f"{s.name.replace(' ', '_')}.parquet"
        if out.exists():
            continue
        try:
            df = d3_power.fetch_point(s.lat, s.lon, args.start, args.end)
            df.to_parquet(out)
            print(f"{s.name:<20} {len(df):>6} days  NaN {df.isna().mean().mean():.1%}")
        except Exception as exc:  # keep going, report at the end
            failed.append(s.name)
            print(f"{s.name:<20} FAILED: {exc}")
        time.sleep(args.pause)

    files = sorted(D3_DIR.glob("*.parquet"))
    provenance.record(
        "D3",
        source=d3_power.URL,
        licence="NASA open data; cite NASA Langley Research Center POWER Project",
        files=files,
        note="MERRA-2 derived, ~0.5 deg native resolution, community=AG, point queries at "
        "station coordinates. PS is kPa. Day boundaries follow the POWER time standard.",
        extra={
            "parameters": d3_power.PARAMETERS,
            "start": args.start,
            "end": args.end,
            "stations": len(files),
            "failed": failed,
        },
    )
    print(f"\nD3: {len(files)} station files, failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
