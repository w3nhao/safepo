print-%  : ; @echo $* = $($*)
PROJECT_NAME   = safepo
COPYRIGHT      = "PKU Alignment Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=

.PHONY: default
default: install

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(1) --upgrade)
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(2) --upgrade)

# Installations

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools
	$(PYTHON) -m pip install -e .

install-e: install-editable  # alias

docs-install:
	$(call check_pip_install_extra,pydocstyle,pydocstyle[toml])
	$(call check_pip_install,doc8)
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install_extra,sphinxcontrib-spelling,sphinxcontrib-spelling pyenchant)
	$(PYTHON) -m pip install -r docs/requirements.txt

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

# Benchmark

multi-benchmark:
	cd safepo/multi_agent && $(PYTHON) benchmark.py --total-steps 10000000 --experiment benchmark

single-benchmark:
	cd safepo/single_agent && $(PYTHON) benchmark.py --total-steps 10000000  --experiment benchmark

multi-simple-benchmark:
	cd safepo/multi_agent && $(PYTHON) benchmark.py --total-steps 10000000 --experiment benchmark --tasks \
	 Safety2x4AntVelocity-v0 Safety4x2AntVelocity-v0 \
	 Safety2x3HalfCheetahVelocity-v0 Safety6x1HalfCheetahVelocity-v0 \

single-simple-benchmark:
	cd safepo/single_agent && $(PYTHON) benchmark.py --total-steps 10000000  --experiment benchmark --tasks \
	 SafetyAntVelocity-v1 SafetyHumanoidVelocity-v1 \
	 SafetyPointGoal1-v0 SafetyCarButton1-v0 \

multi-test-benchmark:
	cd safepo/multi_agent && $(PYTHON) benchmark.py --total-steps 2000 --experiment benchmark --num-envs 1 --tasks \
	 Safety2x4AntVelocity-v0 Safety4x2AntVelocity-v0 \
	 Safety2x3HalfCheetahVelocity-v0 Safety6x1HalfCheetahVelocity-v0 \

single-test-benchmark:
	cd safepo/single_agent && $(PYTHON) benchmark.py --total-steps 2000  --experiment benchmark --num-envs 1 --steps-per-epoch 1000 --tasks \
	 SafetyAntVelocity-v1 SafetyHumanoidVelocity-v1 \
	 SafetyPointGoal1-v0 SafetyCarButton1-v0 \

plot:
	cd safepo && $(PYTHON) plot.py --logdir ./runs/benchmark

eval:
	cd safepo && $(PYTHON) evaluate.py --benchmark-dir ./runs/benchmark

simple-benchmark: install-editable multi-simple-benchmark single-simple-benchmark plot eval

test-benchmark: install-editable multi-test-benchmark single-test-benchmark plot eval

benchmark: install-editable multi-benchmark single-benchmark plot eval

pytest: pytest-install
	cd tests &&  \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="../safepo" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) . 

# Documentation

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	$(PYTHON) -m sphinx_autobuild -b spelling docs/source docs/build