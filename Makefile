#########
# BUILD #
#########
.PHONY: develop build install

develop:  ## install dependencies and build library
	uv pip install -e .[develop]

requirements:  ## install prerequisite python build requirements
	uv pip install -r pyproject.toml --extra develop

build:  ## build the python library
	python -m build -n

install:  ## install library
	uv pip install .

#########
# LINTS #
#########
.PHONY: lint-py lint-docs fix-py fix-docs lint lints fix format

lint-py:  ## lint python with ruff
	python -m ruff check dau_sim
	python -m ruff format --check dau_sim

lint-docs:  ## lint docs with mdformat and codespell
	python -m mdformat --check README.md docs/src
	python -m codespell_lib README.md docs/src

fix-py:  ## autoformat python code with ruff
	python -m ruff check --fix dau_sim
	python -m ruff format dau_sim

fix-docs:  ## autoformat docs with mdformat and codespell
	python -m mdformat README.md docs/src
	python -m codespell_lib --write README.md docs/src

lint: lint-py lint-docs  ## run all linters
lints: lint
fix: fix-py fix-docs  ## run all autoformatters
format: fix

################
# Other Checks #
################
.PHONY: check-dist check-types checks check

check-dist:  ## check python sdist and wheel with check-dist
	check-dist -v

check-types:  ## check python types with ty
	ty check --python $$(which python)

checks: check-dist

# Alias
check: checks

#########
# TESTS #
#########
.PHONY: test coverage tests

test:  ## run python tests
	python -m pytest -v dau_sim/tests

coverage:  ## run tests and collect test coverage
	python -m pytest -v dau_sim/tests --cov=dau_sim --cov-report term-missing --cov-report xml

# Alias
tests: test

##############
# BENCHMARKS #
##############
.PHONY: benchmark benchmark-local benchmark-local-quick benchmark-cross-quick benchmark-cross-runtime benchmark-compare

BENCHMARK_DIR := dau_sim/benchmarks
BENCHMARK_RESULTS_DIR := $(BENCHMARK_DIR)/results
BENCHMARK_FILES := $(BENCHMARK_DIR)/bench_*.py

benchmark: benchmark-local  ## run full local pytest-benchmark suite

benchmark-local:  ## run local pytest-benchmark suite
	mkdir -p $(BENCHMARK_RESULTS_DIR)
	python -m pytest $(BENCHMARK_FILES) -v --benchmark-only --benchmark-columns=mean,stddev,median,iqr,rounds --benchmark-save=local --benchmark-storage=file://$(BENCHMARK_RESULTS_DIR) --benchmark-json=$(BENCHMARK_RESULTS_DIR)/local.json

benchmark-local-quick:  ## run quick local pytest-benchmark suite
	mkdir -p $(BENCHMARK_RESULTS_DIR)
	python -m pytest $(BENCHMARK_FILES) -v --benchmark-only --benchmark-min-rounds=1 --benchmark-max-time=0.02 --benchmark-columns=mean,stddev,median,rounds --benchmark-save=local-quick --benchmark-storage=file://$(BENCHMARK_RESULTS_DIR) --benchmark-json=$(BENCHMARK_RESULTS_DIR)/local-quick.json

benchmark-cross-quick:  ## run quick cross-simulator benchmarks only
	mkdir -p $(BENCHMARK_RESULTS_DIR)
	python -m pytest $(BENCHMARK_DIR)/bench_cross_simulators.py -v --benchmark-only --benchmark-min-rounds=1 --benchmark-max-time=0.02 --benchmark-columns=mean,stddev,median,rounds --benchmark-save=cross-quick --benchmark-storage=file://$(BENCHMARK_RESULTS_DIR) --benchmark-json=$(BENCHMARK_RESULTS_DIR)/cross-quick.json

benchmark-cross-runtime:  ## run cross-simulator benchmarks with larger cycle count for runtime-dominant comparisons
	mkdir -p $(BENCHMARK_RESULTS_DIR)
	DAU_BENCH_CYCLES=500000 python -m pytest $(BENCHMARK_DIR)/bench_cross_simulators.py -v --benchmark-only --benchmark-min-rounds=3 --benchmark-columns=mean,stddev,median,rounds --benchmark-save=cross-runtime-500k --benchmark-storage=file://$(BENCHMARK_RESULTS_DIR) --benchmark-json=$(BENCHMARK_RESULTS_DIR)/cross-runtime-500k.json

benchmark-compare:  ## compare latest benchmark run against previous saved run
	python -m pytest $(BENCHMARK_FILES) -v --benchmark-only --benchmark-compare --benchmark-storage=file://$(BENCHMARK_RESULTS_DIR)

###########
# VERSION #
###########
.PHONY: show-version patch minor major

show-version:  ## show current library version
	@bump-my-version show current_version

patch:  ## bump a patch version
	@bump-my-version bump patch

minor:  ## bump a minor version
	@bump-my-version bump minor

major:  ## bump a major version
	@bump-my-version bump major

########
# DIST #
########
.PHONY: dist dist-build dist-sdist dist-local-wheel publish

dist-build:  # build python dists
	python -m build -w -s

dist-check:  ## run python dist checker with twine
	python -m twine check dist/*

dist: clean dist-build dist-check  ## build all dists

publish: dist  ## publish python assets

#########
# CLEAN #
#########
.PHONY: deep-clean clean

deep-clean: ## clean everything from the repository
	git clean -fdx

clean: ## clean the repository
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info

############################################################################################

.PHONY: help

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
