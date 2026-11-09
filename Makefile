# India Rainfall Benchmark - pipeline entry points
.PHONY: help install data features train evaluate figures readme test lint clean

PY := python

help:
	@echo "install   - install package + dev extras"
	@echo "data      - fetch D1-D5 (IMD, IMDAA, NASA POWER, ERA5, indices)"
	@echo "features  - build the engineered feature table"
	@echo "train     - train the model roster"
	@echo "evaluate  - score models -> results/metrics.csv"
	@echo "figures   - regenerate every paper figure"
	@echo "readme    - refresh auto-generated README sections"
	@echo "test lint - checks"

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/01_fetch_d1_imd.py
	$(PY) scripts/02_fetch_d2_imdaa.py
	$(PY) scripts/03_fetch_d3_power.py
	$(PY) scripts/04_fetch_d4_era5.py
	$(PY) scripts/05_fetch_d5_indices.py
	$(PY) scripts/06_build_dataset.py

features:
	$(PY) scripts/07_build_features.py

train:
	$(PY) scripts/08_train.py

evaluate:
	$(PY) scripts/09_evaluate.py

figures:
	$(PY) scripts/10_figures.py

readme:
	-$(PY) scripts/update_readme.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy src

clean:
	rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
