"""Stage 13 (Phase 2): screen D1 labels for multi-year level shifts.

For each station, the 5-year mean of annual rainfall (1979-2024) is compared
with the station median of those blocks. A ratio far from 1 means the gridded
label is unstable (cell dominated by sparse gauges), not that the weather
changed. Output: reports/eda/label_stability.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from indiarain.config import ROOT

OUT = ROOT / "reports" / "eda"


def main() -> int:
    a = pd.read_csv(OUT / "annual_total_mm.csv", index_col=0).loc[1979:2024]
    blocks = a.groupby(a.index // 5 * 5).mean()
    res = pd.DataFrame({
        "median_block_mm": blocks.median(),
        "max_ratio": blocks.max() / blocks.median(),
        "min_ratio": blocks.min() / blocks.median(),
    })
    res["flag"] = (res["max_ratio"] > 1.8) | (res["min_ratio"] < 0.4)
    res = res.round(2).sort_values("max_ratio", ascending=False)
    res.to_csv(OUT / "label_stability.csv")
    print(res.head(10).to_string())
    print("flagged:", list(res.index[res["flag"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
