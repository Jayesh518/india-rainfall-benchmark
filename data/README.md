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
| D5 | ENSO / IOD / MJO indices | NOAA PSL, Australian BoM | decades | n/a | Public domain | `05_fetch_d5_indices.py` |

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
