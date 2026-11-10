"""Stage 02: D2 (IMDAA reanalysis), the primary predictor source.

IMDAA is not an open download. It needs an account on the NCMRWF Research Data
Service (https://rds.ncmrwf.gov.in), or the pre-curated BharatBench copy on
Kaggle (needs a Kaggle login). Neither can be scripted without the owner's
credentials, so this stage only checks for them and records the status honestly
in the provenance manifest. Nothing is guessed or substituted.

Licence: CC BY-NC-SA 4.0 (non-commercial, share-alike).
"""

from __future__ import annotations

import os
import sys

from indiarain.config import DATA_RAW
from indiarain.data import provenance

D2_DIR = DATA_RAW / "d2_imdaa"


def main() -> int:
    present = [f for f in D2_DIR.rglob("*") if f.is_file()] if D2_DIR.exists() else []
    if present:
        provenance.record(
            "D2",
            source="https://rds.ncmrwf.gov.in/datasets (or BharatBench, arXiv:2405.07534)",
            licence="CC BY-NC-SA 4.0 (non-commercial, share-alike); attribute NCMRWF/MoES",
            files=present,
            verified=False,
            note="Files placed manually; origin and extent not verified by a script.",
        )
        print(f"D2: {len(present)} manual file(s) recorded as unverified.")
        return 0

    have_login = bool(os.environ.get("NCMRWF_USERNAME") and os.environ.get("NCMRWF_PASSWORD"))
    print("D2 (IMDAA) NOT ACQUIRED.")
    print(f"  NCMRWF credentials in .env: {'yes' if have_login else 'no'}")
    print("  Options: (a) register at https://rds.ncmrwf.gov.in and download by hand into")
    print(f"           {D2_DIR}; (b) get BharatBench from Kaggle (maslab/bharatbench).")
    print("  Phase-1 consequence: E1 (main benchmark) cannot run until D2 exists; E2 can")
    print("  run on D3 vs D4 only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
