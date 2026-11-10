"""Stage 05: fetch D5 climate-mode indices (Nino 3.4, SOI, DMI monthly; OMI daily).

Outputs
    data/raw/d5_indices/monthly.csv   month, nino34, soi, dmi
    data/raw/d5_indices/omi.csv       date, omi_pc1, omi_pc2, omi_amp
    data/manifest.json                provenance entry "D5"
"""

from __future__ import annotations

import sys

from indiarain.config import DATA_RAW
from indiarain.data import d5_indices, provenance

D5_DIR = DATA_RAW / "d5_indices"


def main() -> int:
    D5_DIR.mkdir(parents=True, exist_ok=True)
    monthly = d5_indices.fetch_monthly()
    omi = d5_indices.fetch_omi()
    monthly.to_csv(D5_DIR / "monthly.csv")
    omi.to_csv(D5_DIR / "omi.csv")

    spans = {
        c: [str(monthly[c].dropna().index.min().date()), str(monthly[c].dropna().index.max().date())]
        for c in monthly
    }
    spans["omi"] = [str(omi.dropna().index.min().date()), str(omi.dropna().index.max().date())]
    provenance.record(
        "D5",
        source="https://psl.noaa.gov (nina34.anom.data, soi.data, dmi.had.long.data, "
        "mjo/mjoindex/omi.1x.txt)",
        licence="NOAA PSL public data; cite NOAA Physical Sciences Laboratory",
        files=sorted(D5_DIR.glob("*.csv")),
        note="DEVIATION: MJO is the NOAA OMI index, not the BoM RMM index named in the plan, "
        "because bom.gov.au blocks automated downloads.",
        extra={"coverage": spans},
    )
    for k, (a, b) in spans.items():
        print(f"{k:<7} {a} -> {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
