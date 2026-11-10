"""Stage 12 (Phase 2): onset summary, wet/dry spells, extreme catalogue, trends.

Reads the base table (1979-2024) and the full D1 station record (1901-2024).
Outputs go to ``reports/eda`` (git-ignored).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from indiarain.config import DATA_INTERIM, HEAVY_RAIN_MM, RAINY_DAY_MM, ROOT, STATIONS

OUT = ROOT / "reports" / "eda"


def spells(rainy: np.ndarray) -> tuple[list[int], list[int]]:
    wet, dry, run, cur = [], [], 0, None
    for r in rainy:
        if np.isnan(r):
            cur, run = None, 0
            continue
        if r == cur:
            run += 1
        else:
            if cur == 1:
                wet.append(run)
            elif cur == 0:
                dry.append(run)
            cur, run = r, 1
    return wet, dry


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    zone = {s.name: s.zone for s in STATIONS}
    full = pd.read_parquet(DATA_INTERIM / "d1_station_rain.parquet")
    full["date"] = pd.to_datetime(full["date"])
    full["year"] = full["date"].dt.year
    full["month"] = full["date"].dt.month
    full["rainy"] = np.where(full["rain_mm"].isna(), np.nan, (full["rain_mm"] >= RAINY_DAY_MM) * 1.0)

    # onset proxy summary (1979-2024 table made by stage 11)
    on = pd.read_csv(OUT / "monsoon_onset_doy_proxy.csv", index_col=0)
    on_sum = pd.DataFrame({"median_doy": on.median(), "sd_days": on.std(),
                           "never_share": on.isna().mean()}).round(2)
    on_sum["median_date"] = pd.to_datetime(on_sum["median_doy"].round() - 1, unit="D",
                                           origin="2001-01-01").dt.strftime("%d %b")
    on_sum.to_csv(OUT / "onset_summary.csv")

    # wet / dry spells in JJAS (monsoon season), 1901-2024
    rows = []
    for st, g in full.groupby("station"):
        jjas = g[g["month"].between(6, 9)]
        wet, dry = [], []
        for _, gy in jjas.groupby("year"):
            w, d = spells(gy.sort_values("date")["rainy"].to_numpy())
            wet += w
            dry += d
        rows.append({"station": st, "zone": zone[st],
                     "wet_mean": np.mean(wet) if wet else np.nan,
                     "dry_mean": np.mean(dry) if dry else np.nan,
                     "dry_p90": np.percentile(dry, 90) if dry else np.nan,
                     "dry_max": max(dry) if dry else np.nan})
    pd.DataFrame(rows).round(2).to_csv(OUT / "monsoon_spells_1901_2024.csv", index=False)

    # extreme catalogue: top-5 days and heavy-day counts by decade (full record)
    top = (full.dropna(subset=["rain_mm"]).sort_values("rain_mm", ascending=False)
           .groupby("station").head(5)[["station", "date", "rain_mm"]])
    top.sort_values(["station", "rain_mm"], ascending=[True, False]).to_csv(
        OUT / "extreme_top5_per_station.csv", index=False)
    full["decade"] = (full["year"] // 10) * 10
    heavy = full.assign(h=(full["rain_mm"] >= HEAVY_RAIN_MM)).groupby(["decade", "station"])["h"].sum()
    heavy.unstack(1).to_csv(OUT / "heavy_days_by_decade.csv")

    # trends (Mann-Kendall via Kendall tau on annual total; Sen slope), 1901-2024
    tr = []
    for st, g in full.groupby("station"):
        a = g.groupby("year")["rain_mm"].agg(["sum", "count"])
        a = a[a["count"] >= 360]["sum"]
        tau, p = stats.kendalltau(a.index, a.values)
        sen = stats.theilslopes(a.values, a.index)[0]
        tr.append({"station": st, "years": len(a), "tau": tau, "p": p, "sen_mm_per_decade": sen * 10})
    trd = pd.DataFrame(tr).round(4)
    trd.to_csv(OUT / "annual_trend_1901_2024.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(8, 8))
    o = on_sum.sort_values("median_doy")
    ax.errorbar(o["median_doy"], range(len(o)), xerr=o["sd_days"], fmt="o", ms=3)
    ax.set_yticks(range(len(o)), o.index)
    ax.set_xlabel("onset proxy (day of year; first 5-day total >= 40 mm after 1 May)")
    fig.tight_layout()
    fig.savefig(OUT / "onset_by_station.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dec = full.groupby("decade")["rain_mm"].apply(lambda x: (x >= HEAVY_RAIN_MM).sum() / x.size * 365.25)
    dec.plot(kind="bar", ax=ax)
    ax.set_ylabel("heavy-rain days per station-year")
    ax.set_title(f"Days >= {HEAVY_RAIN_MM} mm by decade (all stations, IMD)")
    fig.tight_layout()
    fig.savefig(OUT / "heavy_days_by_decade.png", dpi=130)
    plt.close(fig)

    print("onset (earliest/latest median):")
    print(on_sum.sort_values("median_doy").iloc[[0, 1, 2, -3, -2, -1]].to_string())
    print("\nsignificant annual trends (p<0.05):")
    print(trd[trd["p"] < 0.05].sort_values("sen_mm_per_decade").to_string(index=False))
    print("\nheavy-day rate per station-year by decade:\n", dec.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
