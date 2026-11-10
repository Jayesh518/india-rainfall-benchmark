"""Stage 07 (Phase 3): build the leak-safe modelling tables.

Outputs data/processed/features_lagged.parquet and features_sameday.parquet.
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import DATA_PROCESSED
from indiarain.features.build import build, feature_columns


def main() -> int:
    base = pd.read_parquet(DATA_PROCESSED / "india_rain_base.parquet")
    for mode in ("lagged", "sameday"):
        df = build(base, mode)
        # 1981 start: POWER begins then; earlier rows have no predictors
        df = df[df["date"] >= "1981-02-01"].dropna(subset=["y"])
        out = DATA_PROCESSED / f"features_{mode}.parquet"
        df.to_parquet(out, index=False)
        print(f"{mode}: {len(df):,} rows, {len(feature_columns(df))} features, "
              f"rain rate {df['y'].mean():.3f} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
