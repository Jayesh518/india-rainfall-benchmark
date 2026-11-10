"""Stage 11 (Phase 2): exploratory analysis and monsoon climatology.

Uses only the base table, so it can be re-run as D4 grows. Writes tables and
figures to ``reports/eda`` (git-ignored; derived from licensed data).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indiarain.config import DATA_PROCESSED, RAINY_DAY_MM, ROOT, STATIONS

OUT = ROOT / "reports" / "eda"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROCESSED / "india_rain_base.parquet")
    zone = {s.name: s.zone for s in STATIONS}
    df["zone"] = df["station"].map(zone)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["rainy"] = (df["d1_rain_mm"] >= RAINY_DAY_MM).astype(float)
    df.loc[df["d1_rain_mm"].isna(), "rainy"] = np.nan

    # 1. class balance by station x month
    bal = df.pivot_table(index="station", columns="month", values="rainy", aggfunc="mean")
    bal.round(4).to_csv(OUT / "rainy_share_station_month.csv")
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(bal.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(bal)), bal.index)
    ax.set_xticks(range(12), list("JFMAMJJASOND"))
    ax.set_title(f"Share of days with rain >= {RAINY_DAY_MM} mm (IMD, 1979-2024)")
    fig.colorbar(im, ax=ax, label="share of days")
    fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=130)
    plt.close(fig)

    # 2. annual totals and trend per station
    ann = df.groupby(["station", "year"])["d1_rain_mm"].sum().unstack(0)
    ann.to_csv(OUT / "annual_total_mm.csv")
    summary = pd.DataFrame({
        "zone": pd.Series(zone),
        "mean_annual_mm": ann.mean(),
        "cv_annual": ann.std() / ann.mean(),
        "rainy_share": df.groupby("station")["rainy"].mean(),
        "heavy_days_per_yr": df.assign(h=(df["d1_rain_mm"] >= 64.5)).groupby("station")["h"].sum()
        / df["year"].nunique(),
        "max_daily_mm": df.groupby("station")["d1_rain_mm"].max(),
    }).dropna(subset=["mean_annual_mm"]).round(3)
    summary.to_csv(OUT / "station_summary.csv")

    # 3. monsoon onset proxy: first day of year with 5-day rain total >= 40 mm after 1 May
    onset = {}
    for st, g in df.groupby("station"):
        rows = []
        for y, gy in g.groupby("year"):
            gy = gy.set_index("date")["d1_rain_mm"]
            r5 = gy.rolling(5).sum()
            cand = r5[(r5.index >= f"{y}-05-01") & (r5 >= 40)]
            rows.append(cand.index[0].day_of_year if len(cand) else np.nan)
        onset[st] = rows
    on = pd.DataFrame(onset, index=sorted(df["year"].unique()))
    on.to_csv(OUT / "monsoon_onset_doy_proxy.csv")

    # 4. missingness per source
    miss = {}
    for p in ("d1_", "d3_", "d4_", "d5_"):
        cols = [c for c in df.columns if c.startswith(p)]
        if cols:
            miss[p] = df[cols].isna().groupby(df["station"]).mean().mean(axis=1)
    pd.DataFrame(miss).round(4).to_csv(OUT / "missingness_by_source.csv")

    # 5. source agreement for rain: IMD vs POWER (all stations) and ERA5 (where present)
    agree = []
    for st, g in df.groupby("station"):
        row = {"station": st}
        for name, col in (("power", "d3_PRECTOTCORR"), ("era5", "d4_precipitation_sum")):
            if col in g and g[col].notna().sum() > 365:
                s = g[["d1_rain_mm", col]].dropna()
                row[f"{name}_corr"] = s.corr().iloc[0, 1]
                row[f"{name}_pod"] = ((s.iloc[:, 1] >= 2.5) & (s.iloc[:, 0] >= 2.5)).sum() / max(
                    (s.iloc[:, 0] >= 2.5).sum(), 1)
                row[f"{name}_bias_mm_per_day"] = (s.iloc[:, 1] - s.iloc[:, 0]).mean()
                # lag check for the day-boundary question: corr of IMD(t) with source(t+k)
                for k in (-1, 1):
                    sh = g[col].shift(-k)
                    row[f"{name}_corr_lag{k:+d}"] = g["d1_rain_mm"].corr(sh)
        agree.append(row)
    pd.DataFrame(agree).round(3).to_csv(OUT / "source_agreement.csv", index=False)

    # 6. seasonal cycle by zone
    clim = df.groupby(["zone", "month"])["d1_rain_mm"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clim.plot(ax=ax, marker="o", ms=3)
    ax.set_xlabel("month")
    ax.set_ylabel("mean daily rain (mm)")
    ax.set_title("Seasonal cycle by zone (IMD)")
    fig.tight_layout()
    fig.savefig(OUT / "seasonal_cycle_by_zone.png", dpi=130)
    plt.close(fig)

    print(summary.to_string())
    print("\nsource agreement (head):")
    print(pd.read_csv(OUT / "source_agreement.csv").head(10).to_string())
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
