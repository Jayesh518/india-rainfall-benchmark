"""Offline tests for the data layer: parsers, grid matching and provenance.

No network access is needed; fixtures reproduce the real payload shapes seen
from each provider.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from indiarain.config import Station
from indiarain.data import d3_power, d4_openmeteo, d5_indices, provenance
from indiarain.data.d1_imd import MAX_MATCH_KM, haversine_km, match_cells

# --------------------------------------------------------------------------- #
# D1: grid matching
# --------------------------------------------------------------------------- #


def test_haversine_known_distance():
    # One degree of latitude is ~111.2 km.
    d = haversine_km(10.0, 70.0, np.array([11.0]), np.array([70.0]))
    assert d[0] == pytest.approx(111.2, abs=0.5)


def _grid():
    lats = np.arange(10.0, 14.0, 0.25)
    lons = np.arange(70.0, 74.0, 0.25)
    vf = np.ones((lats.size, lons.size))
    return vf, lats, lons


def test_station_on_land_matches_nearby_cell():
    vf, lats, lons = _grid()
    s = Station("Here", 12.1, 72.1, "X", "z")
    (m,) = match_cells(vf, lats, lons, (s,))
    assert m.accepted and m.distance_km is not None and m.distance_km < MAX_MATCH_KM


def test_station_over_gap_is_dropped_not_borrowed():
    """Port Blair case: the grid has no data near the station."""
    vf, lats, lons = _grid()
    vf[:, lons > 71.0] = 0.0  # data only in the west strip
    s = Station("Island", 12.0, 73.5, "X", "z")
    (m,) = match_cells(vf, lats, lons, (s,))
    assert not m.accepted
    assert m.distance_km is not None and m.distance_km > MAX_MATCH_KM


def test_station_with_no_data_anywhere():
    vf, lats, lons = _grid()
    (m,) = match_cells(np.zeros_like(vf), lats, lons, (Station("Void", 12.0, 72.0, "X", "z"),))
    assert not m.accepted and m.distance_km is None


def test_low_coverage_cell_is_not_usable():
    vf, lats, lons = _grid()
    vf[:] = 0.5  # every cell below MIN_VALID_FRACTION
    (m,) = match_cells(vf, lats, lons, (Station("Gappy", 12.0, 72.0, "X", "z"),))
    assert not m.accepted


# --------------------------------------------------------------------------- #
# D3 / D4 payload parsing
# --------------------------------------------------------------------------- #


def test_power_parse_turns_fill_value_into_nan():
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"20240101": 24.68, "20240102": -999.0},
                "PRECTOTCORR": {"20240101": 0.0, "20240102": 1.5},
            }
        }
    }
    df = d3_power.parse(payload)
    assert list(df.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert np.isnan(df.loc["2024-01-02", "T2M"])
    assert df.loc["2024-01-02", "PRECTOTCORR"] == 1.5


def test_openmeteo_parse():
    payload = {
        "daily": {
            "time": ["2024-07-01", "2024-07-02"],
            "precipitation_sum": [11.1, 15.4],
            "surface_pressure_mean": [1001.6, 1001.2],
        }
    }
    df = d4_openmeteo.parse(payload)
    assert df.index.name == "date"
    assert df["precipitation_sum"].tolist() == [11.1, 15.4]
    assert "time" not in df.columns


def test_units_are_documented_for_every_requested_variable():
    assert d3_power.PARAMETERS["PS"] == "kPa", "POWER pressure is kPa; ERA5 is hPa"
    assert d4_openmeteo.DAILY["surface_pressure_mean"] == "hPa"


# --------------------------------------------------------------------------- #
# D5: index parsing
# --------------------------------------------------------------------------- #

PSL_SAMPLE = """ 1948        2026
 1948    -0.43   -0.11    0.12   -0.00    0.43   -0.06    0.00   -0.22   -0.47   -0.95   -1.01    0.12
 1949    -0.38    0.07  -99.99   -0.22    0.03   -1.14   -0.48   -0.50   -0.97   -0.81   -1.64   -1.35
  NINA34
 5N-5S 170W-120W
"""


def test_psl_monthly_parses_table_and_skips_footer():
    s = d5_indices.parse_psl_monthly(PSL_SAMPLE, "nino34")
    assert len(s) == 24
    assert s.loc["1948-01-01"] == -0.43
    assert s.index.is_monotonic_increasing


def test_psl_monthly_missing_becomes_nan():
    s = d5_indices.parse_psl_monthly(PSL_SAMPLE, "nino34")
    assert np.isnan(s.loc["1949-03-01"])
    assert s.isna().sum() == 1


def test_omi_parse_handles_fill_values():
    text = "1991 1 1 0.23390 -0.69290 0.73132\n1991 1 2 1.00000e36 1.00000e36 1.00000e36\n"
    df = d5_indices.parse_omi(text)
    assert list(df.columns) == ["omi_pc1", "omi_pc2", "omi_amp"]
    assert df.loc["1991-01-01", "omi_amp"] == pytest.approx(0.73132)
    assert df.loc["1991-01-02"].isna().all()


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_md5_matches_known_digest(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello")
    assert provenance.md5_file(f) == "5d41402abc4b2a76b9719d911017c592"


def test_record_writes_checksums_and_unverified_flag(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(provenance, "ROOT", root)
    f = root / "data" / "x.csv"
    f.parent.mkdir()
    f.write_text("a,b\n1,2\n")
    manifest = root / "data" / "manifest.json"

    provenance.record(
        "DX", source="s", licence="l", files=[f], verified=False, note="n", path=manifest
    )
    entry = json.loads(manifest.read_text())["DX"]
    assert entry["verified"] is False
    assert entry["files"][0]["path"] == "data/x.csv"
    assert entry["files"][0]["md5"] == provenance.md5_file(f)
    assert entry["fetched_utc"].endswith("Z")


# --------------------------------------------------------------------------- #
# D4 quota handling
# --------------------------------------------------------------------------- #


class _Resp:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_429_raises_quota_exceeded_with_reason(monkeypatch):
    reason = "Hourly API request limit exceeded. Please try again in the next hour."
    monkeypatch.setattr(
        d4_openmeteo.http, "get_once", lambda *a, **k: _Resp(429, {"error": True, "reason": reason})
    )
    with pytest.raises(d4_openmeteo.QuotaExceededError) as exc:
        d4_openmeteo.fetch_point(19.0, 72.8, "2000-01-01", "2009-12-31")
    assert "Hourly" in exc.value.reason


def test_request_weight_scales_with_days():
    ten_years = d4_openmeteo.request_weight(3652)
    assert 250 < ten_years < 270
    assert d4_openmeteo.request_weight(7) == pytest.approx(len(d4_openmeteo.DAILY) / 10)
    # Why the fetch takes days: 34 stations x 44 years vs a 10,000/day allowance.
    total = 34 * d4_openmeteo.request_weight(44 * 365)
    assert total / d4_openmeteo.DAILY_QUOTA > 3
