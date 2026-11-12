# India Rainfall Benchmark

**Next-day rainfall prediction over India — benchmarking skill, calibration and decision value across model families.**

[![CI](https://github.com/Jayesh518/india-rainfall-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Jayesh518/india-rainfall-benchmark/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Data](https://img.shields.io/badge/data-IMD%20%7C%20IMDAA%20%7C%20NASA%20POWER-success)](data/README.md)
[![License](https://img.shields.io/badge/code-MIT-green)](LICENSE)

<!-- AUTO:STATUS:START -->
**Phase:** 1 — Data acquisition (D1–D5) · **Roadmap:** 1 / 16 phases complete (6%)

`█░░░░░░░░░░░░░░░░░░░` 6%
<!-- AUTO:STATUS:END -->

---

## The problem

Published rainfall-ML studies tend to share three weaknesses:

1. **Accuracy on an imbalanced target.** When ~80% of days are dry, a model that always answers "no rain" scores 80%. Many reported figures sit at or below that floor — and the papers cannot tell, because they report no skill score.
2. **Calibration goes unchecked.** A forecast of "70% chance of rain" is only useful if it rains on roughly 70% of such days. Almost nobody verifies this.
3. **The predictor dataset is treated as neutral.** Rankings are published as properties of models, when they may be properties of whichever reanalysis the authors happened to use.

This project addresses all three on **Indian data only**, using **official Government of India sources**, across **46 years** and **35 stations**.

A concrete illustration, from [`tests/test_metrics.py`](tests/test_metrics.py) — a forecaster that *never* predicts rain, on a dataset with 20% rainy days:

| Metric | Score |
|---|---|
| Accuracy | **0.80** |
| Probability of Detection | 0.00 |
| Critical Success Index | 0.00 |
| Heidke Skill Score | 0.00 |
| Equitable Threat Score | 0.00 |

That gap is what this benchmark is built to measure. It is asserted as a unit test, not left as a claim.

## Research questions

| | Question |
|---|---|
| **RQ1** | Across ~40 methods (Markov chains → gradient boosting → transformers → graph networks → weather foundation models), how much do models disagree in *probability*, in *which* days they miss, and in how *calibrated* they are? Does the accuracy winner also win on calibration? |
| **RQ2** | Run one identical pipeline over three independent predictor sources. **Does the model ranking survive a change of reanalysis?** If not, published leaderboards are partly an artefact of predictor choice. |
| **RQ3** | In a downstream reservoir/irrigation decision environment, which forecast produces the best *outcome* — the most accurate model, or the best-calibrated one? |

RQ2 comes nearly free: the multi-source stack has to exist anyway, because the foundation models cannot consume the same inputs as the tabular models (see [E7](#data-routing)).

---

## Data

All sources are India-specific and authoritative. Combined span: **1901–2025**. Full provenance and attribution in [`data/README.md`](data/README.md).

| ID | Source | Authority | Coverage | Resolution | Role |
|---|---|---|---|---|---|
| **D1** | [Gridded daily rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) | India Meteorological Department | **1901–2024** | 0.25°, 135×129 grid | **The label** — rain-gauge based, the national reference product |
| **D2** | [IMDAA reanalysis](https://rds.ncmrwf.gov.in/datasets) | NCMRWF / Ministry of Earth Sciences | 1979–2020 | **12 km**, 63 levels | Primary predictors — the only long, high-resolution, India-specific reanalysis |
| **D3** | [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) | NASA Langley (MERRA-2) | 1981 → **near-real-time** | ~0.5° | Near-real-time predictors; the only source that reaches today |
| **D4** | [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | ECMWF / Copernicus | 1940 → present | 0.25° | Live serving + third predictor source |
| **D5** | ENSO · IOD · MJO indices | NOAA PSL / BoM | decades | — | Teleconnection context |

**Target dataset:** 35 stations × 46 years × 365 days ≈ **550,000 rows**, ~120 engineered features.

### Data routing

Each experiment draws from the source that is *correct* for it. **D1 is always the label**, so every result is comparable.

| Experiment | Predictors | Why this source |
|---|---|---|
| **E0** Baselines (Markov chain, HMM, extreme value theory) | D1 only | Needs rainfall alone — uses the full 124-year record, so climatology and return levels are stable |
| **E1** Main benchmark | D2 + D5 | Richest India-specific predictor set |
| **E2** **Source ablation (RQ2)** | D2 vs D3 vs D4 | Identical pipeline, three sources |
| **E3** Operational model | D3 + D4 | IMDAA stops at 2020 and can never serve live |
| **E5** Nowcasting (ConvLSTM, SmaAt-UNet) | D1 grid | These consume the rainfall *field*, not station points |
| **E6** Graph networks | D2 at nodes | Needs simultaneous multi-site fields |
| **E7** Weather foundation models | **D4 (ERA5)** | Prithvi-WxC / ClimaX / Aurora are pretrained on ERA5 and MERRA-2 — they *cannot* take IMDAA without regridding. **This is what forces the multi-source design.** |

### Target definitions

Following **IMD conventions**, not the Australian ones common in the literature:

- **Binary** — rain if rainfall(t+1) ≥ **2.5 mm** (the IMD "rainy day" standard); 1.0 mm also reported for comparability
- **Regression** — `rainfall_mm(t+1)`, log1p or Tweedie (zero-inflated, heavy-tailed)
- **Ordinal** — IMD's six official intensity categories: No rain · Light · Moderate · Heavy · Very heavy · Extremely heavy
- **Extreme** — ≥ 64.5 mm (~1–2% positives)

Encoded once in [`src/indiarain/config.py`](src/indiarain/config.py) and verified by parametrised tests.

---

## Methods

| Tier | Family |
|---|---|
| **M0** | Persistence · climatology · **Markov chains** · HMM · ARIMA/SARIMA · **extreme value theory (GEV/POT)** · Tweedie GLM · **EMOS** |
| **M1** | Logistic regression · LDA/QDA · k-NN · SVM · Gaussian process · decision tree · **EBM** · random forest · extra trees |
| **M2** | **XGBoost · LightGBM · CatBoost** · **NGBoost** · TabNet · FT-Transformer · **TabPFN-v2** · stacking |
| **M3** | 1D-CNN · TCN · LSTM/GRU · N-BEATS · **TFT** · PatchTST · iTransformer · **DLinear** · Mamba · zero-shot **Chronos / TimesFM / TabPFN-TS** |
| **M4** | ConvLSTM · U-Net · **SmaAt-UNet** · diffusion · GCN/GAT/**STGCN**/Graph-WaveNet · **GraphCast-style mesh** |
| **M5** | **Prithvi-WxC** (frozen encoder + probe) · ClimaX · Aurora · self-supervised pretraining · distillation |
| **M6** | **RL:** adaptive ensemble selection (LinUCB → DQN → PPO → offline CQL) · cost-sensitive alert policy · reservoir decision environment |

**Cross-cutting:** calibration (Platt, isotonic, temperature) · **conformal prediction** · deep ensembles · SHAP/ALE · **PCMCI causal discovery** · AutoGluon as an honesty check.

> **On reinforcement learning:** RL is deliberately *not* used for the prediction step. Next-day rain from today's observations is one-shot supervised learning with abundant labels, so "a DQN predicts rainfall" invites an obvious objection. RL is applied where it is genuinely the right tool: **ensemble selection** (which model to trust today, rewarded by a proper scoring rule) and **downstream decision-making**.

### Evaluation

Meteorological verification scores alongside ML ones — **POD, FAR, CSI, Heidke Skill Score, Equitable Threat Score, Peirce Skill Score, Brier Skill Score**, Murphy's Brier decomposition — plus PR-AUC, ECE and MCE. Implemented and tested in [`src/indiarain/evaluation/metrics.py`](src/indiarain/evaluation/metrics.py).

Validation is **temporal, never random** — a random split on a time series trains on 2016 to predict 2012. Blocked/expanding CV with a ≥30-day purge gap, leave-one-station-out, leave-one-climate-zone-out, and **Friedman → Nemenyi → critical-difference diagrams** for significance. No claim of "A beats B" from a single number.

Three baselines accompany every experiment so no result floats without a floor: **always-dry**, **persistence** (tomorrow = today), and **climatology** (per-station, per-month base rate).

---

## Results

<!-- AUTO:RESULTS:START -->
_No results yet — the benchmark is in development. This table is generated from `results/metrics.csv` by `scripts/update_readme.py` and will populate automatically once the first experiments run._
<!-- AUTO:RESULTS:END -->

---

## Repository structure

```
india-rainfall-benchmark/
├── conf/                  # Hydra configs: data, model, experiment, split
├── data/                  # fetched at runtime, gitignored; README.md holds provenance
├── src/indiarain/
│   ├── config.py          # paths, IMD thresholds and categories, 35 stations
│   ├── data/              # D1-D5 loaders, regridding, schema validation
│   ├── features/          # lags, rolling, physics, monsoon, teleconnections
│   ├── models/            # statistical -> boosting -> sequence -> graph -> foundation
│   ├── evaluation/        # metrics.py: meteorological scores + calibration
│   ├── explain/           # SHAP, ALE, permutation, cross-model attribution
│   ├── viz/               # reliability diagrams, CD diagrams, skill maps
│   └── rl/                # ensemble-selection and reservoir environments
├── scripts/               # numbered pipeline stages
├── tests/                 # metric correctness, schema, no-leakage guards
├── notebooks/             # narrative only, no logic
├── results/               # metrics.csv, predictions, figures
├── api/                   # FastAPI /predict (ONNX, zero training in request path)
└── dashboard/             # Streamlit model-comparison dashboard
```

**Stack:** Hydra · DVC · MLflow · Optuna · pandera · ONNX · FastAPI · Streamlit · Docker

---

## Quickstart

```bash
git clone https://github.com/Jayesh518/india-rainfall-benchmark.git
cd india-rainfall-benchmark

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install

cp .env.example .env     # only D2 (NCMRWF) needs credentials

make data                # fetch D1-D5
make features
make train
make evaluate            # -> results/metrics.csv
make figures
```

Run the tests with `make test`, or `PYTHONPATH=src pytest -q`.

---

## Roadmap

- [x] **0** — Scaffold, metric suite, IMD conventions, CI
- [ ] **1** — Data acquisition (D1–D5)
- [ ] **2** — Exploratory analysis & monsoon climatology
- [ ] **3** — Feature engineering
- [ ] **4** — Statistical baselines (M0)
- [ ] **5** — Main benchmark (M1 + M2)
- [ ] **6** — Predictor-source ablation (RQ2)
- [ ] **7** — Calibration & uncertainty quantification
- [ ] **8** — Model disagreement analysis
- [ ] **9** — Sequence models (M3)
- [ ] **10** — Spatiotemporal & graph models (M4)
- [ ] **11** — Reinforcement learning (M6)
- [ ] **12** — Foundation models & self-supervised pretraining (M5)
- [ ] **13** — Causal discovery & explainability
- [ ] **14** — FastAPI + Streamlit application
- [ ] **15** — Paper & write-up

<!-- AUTO:STATS:START -->
<sub>**Repository:** 63 tracked files · 57 source files · ~4,024 lines of code · 11 commits</sub>
<!-- AUTO:STATS:END -->

---

## Citation

```bibtex
@software{chaudhari_india_rainfall_benchmark,
  author = {Chaudhari, Jayesh},
  title  = {India Rainfall Benchmark: Skill, Calibration and Decision Value
            in Machine Learning for Next-Day Rainfall over India},
  year   = {2026},
  url    = {https://github.com/Jayesh518/india-rainfall-benchmark}
}
```

## Licensing

Code is [MIT](LICENSE). **The data is not** — each source carries its own terms, and
**IMDAA (NCMRWF) is CC BY-NC-SA 4.0: non-commercial, share-alike.**
This repository redistributes **no** source data and no third-party papers — only the
scripts that fetch data. See [`data/README.md`](data/README.md) for required attribution.

## Acknowledgements

Builds on [BharatBench](https://github.com/MASLABnitrkl/BharatBench) (Choudhury, Panda & Mukherjee, NIT Rourkela — [arXiv:2405.07534](https://arxiv.org/abs/2405.07534)) and [IndiaWeatherBench](https://github.com/tung-nd/IndiaWeatherBench) (Nguyen et al., UCLA — [arXiv:2509.00653](https://arxiv.org/abs/2509.00653)), which curated IMDAA for machine learning and established graph-model baselines over India.

<!-- AUTO:UPDATED:START -->
<sub>Auto-generated sections last updated **2026-11-12 09:24 UTC** from commit [`11a347c`](../../commit/11a347c) by `scripts/update_readme.py`.</sub>
<!-- AUTO:UPDATED:END -->
