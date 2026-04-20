# ─────────────────────────────────────────────
#  Tredence Case Study — Makefile
# ─────────────────────────────────────────────
#
#  Quickstart
#  ----------
#  python -m venv venv && source venv/bin/activate
#  make setup      ← installs the package + all dev deps from pyproject.toml
#  make train      ← runs the adaptive pruning experiment
#  make test       ← runs the full pytest suite
#  make lint       ← formats code with black + isort (writes in-place)
#  make clean      ← removes downloaded data, outputs, and caches

.PHONY: setup train test lint clean

# Install the project in editable mode together with all dev dependencies.
# `pip install -e .[dev]` reads pyproject.toml so versions are always in sync
# with what is declared there — no separate requirements.txt needed.
setup:
	pip install -e ".[dev]"

train:
	python -m src.train

test:
	pytest tests/ -v

# Run black then isort so imports are sorted after any reformatting.
lint:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf data/ outputs/ __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache