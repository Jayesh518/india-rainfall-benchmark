"""Stage 06: assemble the base station-day table from the fetched sources.

One row per (station, date). D1 rainfall is the label source. Predictor columns
keep their source prefix (``d3_``, ``d4_``, ``d5_``) and native units so that
nothing is mixed silently; unit harmonisation and feature engineering belong to
Phase 3. ``y_rain_next_mm`` is rainfall on the following day and must never be
used as a feature (it is the only ``y_`` column).

Outputs
    data/processed/india_rain_base.parquet
    data/processed/coverage.csv       per-station rows and missing share per source
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import DATA_INTERIM, DATA_PROCESSED, STUDY_END_YEAR, STUDY_START_YEAR
from indiarain.data import provenance

RAW = DATA_INTERIM.parent / "raw"


def _load_station_dir(folder: str, prefix: str) -> pd.DataFrame:
    frames = []
    for f in sorted((RAW / folder).glob("*.parquet")):
        df = pd.read_parquet(f).add_prefix(prefix)
        df["station"] = f.stem.replace("_", " ")
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build() -> pd.DataFrame:
    d1 = pd.read_parquet(DATA_INTERIM / "d1_station_rain.parquet")
    d1["date"] = pd.to_datetime(d1["date"])
    d1 = d1[(d1.date.dt.year >= STUDY_START_YEAR) & (d1.date.dt.year <= STUDY_END_YEAR)]
    d1 = d1.sort_values(["station", "date"])
    d1["y_rain_next_mm"] = d1.groupby("station")["rain_mm"].shift(-1)

    df = d1.rename(columns={"rain_mm": "d1_rain_mm"})
    for folder, prefix in (("d3_power", "d3_"), ("d4_era5", "d4_")):
        part = _load_station_dir(folder, prefix)
        if not part.empty:
            part = part.rename(columns={f"{prefix}date": "date", "date": "date"})
            df = df.merge(part, on=["station", "date"], how="left")

    indices = RAW / "d5_indices"
    if (indices / "monthly.csv").exists():
        m = pd.read_csv(indices / "monthly.csv", parse_dates=["month"]).add_prefix("d5_")
        df["_month"] = df["date"].values.astype("datetime64[M]")
        df = df.merge(m, left_on="_month", right_on="d5_month", how="left").drop(
            columns=["_month", "d5_month"]
        )
    if (indices / "omi.csv").exists():
        o = pd.read_csv(indices / "omi.csv", parse_dates=["date"])
        df = df.merge(o.rename(columns={c: f"d5_{c}" for c in o.columns if c != "date"}),
                      on="date", how="left")
    return df.sort_values(["station", "date"]).reset_index(drop=True)


def main() -> int:
    df = build()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "india_rain_base.parquet"
    df.to_parquet(out, index=False)

    cov = df.groupby("station").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"))
    for prefix in ("d1_", "d3_", "d4_", "d5_"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            cov[f"{prefix}missing"] = df[cols].isna().groupby(df["station"]).mean().mean(axis=1)
    cov.to_csv(DATA_PROCESSED / "coverage.csv")
    print(cov.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{len(df):,} rows x {df.shape[1]} columns -> {out}")

    provenance.record(
        "BASE",
        source="derived from D1, D3, D4, D5 (see their entries)",
        licence="Derived; inherits the most restrictive source licence. Not redistributed.",
        files=[out],
        note="Base station-day table. Native units kept; no features engineered yet.",
        extra={"rows": len(df), "columns": list(df.columns), "stations": int(cov.shape[0])},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
