.PHONY: lint typecheck test test-golden test-golden-offline ci ci-pr ci-full test-then-ci dev install install-tool wiki-lint

# ── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  RUN := uv run
else
  RUN := .venv/bin/
endif

# ── Code Quality ─────────────────────────────────────────────────────────────

lint:  ## Ruff lint check (no auto-fix)
	$(RUN) ruff check src/ tests/

wiki-lint:  ## Validate docs/wiki structure and metadata
	python3 scripts/wiki_lint.py

lint-fix:  ## Ruff lint with auto-fix
	$(RUN) ruff check --fix src/ tests/

typecheck:  ## Mypy strict type check
	$(RUN) mypy src/redthread/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:  ## Unit tests (no API calls)
	PYTHONPATH=src REDTHREAD_DRY_RUN=true \
	$(RUN) pytest tests/ \
		--ignore=tests/test_golden_dataset.py \
		-v --tb=short

test-golden:  ## Golden Dataset regression (requires OPENAI_API_KEY)
	PYTHONPATH=src \
	$(RUN) pytest tests/test_golden_dataset.py \
		-v --tb=short

test-golden-offline:  ## Sealed golden regression matching GitHub Actions
	PYTHONPATH=src REDTHREAD_DRY_RUN=true \
	$(RUN) pytest tests/test_golden_dataset.py \
		-v --tb=short

# ── CI ────────────────────────────────────────────────────────────────────────

ci: lint typecheck test  ## Fast local CI: lint + typecheck + unit tests

ci-pr: lint typecheck test test-golden-offline  ## Full local PR CI mirror: lint + typecheck + unit tests + sealed golden regression

ci-full: lint typecheck test test-golden  ## Full CI including live golden regression

test-then-ci:  ## Run focused pytest first, then local PR CI mirror. Usage: make test-then-ci PYTEST_ARGS="tests/test_file.py -q"
	./scripts/test_then_ci.sh $(PYTEST_ARGS)

# ── Project Setup ─────────────────────────────────────────────────────────────

dev:  ## Install with dev dependencies (editable mode)
ifdef UV
	uv sync --extra dev
else
	pip install -e ".[dev]"
endif

install:  ## Install in editable mode (no dev extras)
ifdef UV
	uv sync
else
	pip install -e .
endif

install-tool:  ## Install global `redthread` command with uv and run bootstrap flow
	bash scripts/install_redthread.sh

# ── Dashboard & Monitoring ────────────────────────────────────────────────────

dashboard:  ## View campaign history dashboard
	$(RUN) redthread dashboard

monitor:  ## Start the Security Guard daemon
	$(RUN) redthread monitor start

status:  ## Show current ASI health status
	$(RUN) redthread monitor status

# ── Local Models & Inference ──────────────────────────────────────────────────

llama-cpp-setup:  ## Clone and build llama.cpp from source (macOS Metal optimized)
	@if [ ! -d "vendor/llama.cpp" ]; then \
		mkdir -p vendor && git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp; \
	fi
	cd vendor/llama.cpp && make -j

model-setup:  ## Pull Gemma 4 E4B model via Ollama
	ollama pull gemma4:e4b
