# Data provenance

No data is committed to this repository. Every file below is fetched by a
script in `scripts/` and lands in `data/raw/`, which is gitignored.

Record the download date, row count and md5 for each file as you acquire it,
so results stay reproducible.

| ID | Dataset | Authority | Coverage | Resolution | Licence | Fetched by |
|----|---------|-----------|----------|------------|---------|------------|
| D1 | Gridded daily rainfall | India Meteorological Department, Pune | 1901-2024 | 0.25 deg, 135x129 grid | IMD terms; cite IMD | `01_fetch_d1_imd.py` |
| D2 | IMDAA regional reanalysis | NCMRWF, Ministry of Earth Sciences | 1979-2020 | 12 km, 63 levels | **CC BY-NC-SA 4.0** | `02_fetch_d2_imdaa.py` |
| D3 | POWER daily point | NASA Langley (MERRA-2) | 1981-present | ~0.5 deg | NASA open data | `03_fetch_d3_power.py` |
| D4 | ERA5 archive + forecast | ECMWF / Copernicus via Open-Meteo | 1940-present | 0.25 deg | Free non-commercial | `04_fetch_d4_era5.py` |
| D5 | ENSO / IOD / MJO indices | NOAA PSL | Nino3.4 1948-, SOI 1951-, DMI 1870-, OMI 1991- | n/a | Public domain | `05_fetch_d5_indices.py` |

## Required attribution

- **D1** - India Meteorological Department, Pune. High-resolution (0.25 deg)
  long-period (1901-2024) daily gridded rainfall over India.
- **D2** - IMDAA reanalysis, NCMRWF, Ministry of Earth Sciences, Government of
  India, produced under the National Monsoon Mission with the UK Met Office
  and IMD. **Non-commercial use only; share-alike.**
- **D3** - NASA Langley Research Center POWER Project.
- **D4** - Zippenfenig, P. (2023). Open-Meteo.com Weather API.
  DOI 10.5281/ZENODO.7970649. Plus: "Generated using Copernicus Climate
  Change Service information."

## Label convention

The target is built from **D1 only**, so every experiment shares one ground
truth. A "rainy day" is >= 2.5 mm, the IMD standard -- not the 1.0 mm
Australian convention used by much of the rainfall-ML literature. See
`src/indiarain/config.py`.

## Acquisition status and deviations (Phase 1, 2026-11-10)

Machine-readable record: [`manifest.json`](manifest.json) (source, licence,
fetch time, size and md5 of every file, and a `verified` flag).

- **D2 (IMDAA) is not acquired.** It needs an NCMRWF portal account or the
  BharatBench copy on Kaggle; neither can be scripted without credentials.
  Until it is present, the main benchmark (E1) cannot run on IMDAA and the
  source ablation (E2) compares D3 against D4 only.
- **Port Blair is dropped.** The IMD gauge grid has no valid data over the
  Andaman Islands, so the station has no label. Stations are matched to the
  nearest grid cell that has data and are rejected beyond 40 km; the accepted
  cell and distance for each station are in the manifest.
- **MJO index changed.** The plan named the BoM RMM index, but bom.gov.au
  blocks automated downloads. The NOAA PSL OMI index is used instead; it starts
  in 1991, so MJO values are missing for 1979-1990.
- **Units are kept native.** POWER pressure is kPa and wind m/s; ERA5 pressure
  is hPa and wind km/h. Harmonisation happens in feature engineering.
- **Day alignment is unverified.** How the IMD rainfall day lines up with the
  POWER and Open-Meteo day definitions is checked in Phase 2 (EDA).
