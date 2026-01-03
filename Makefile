.PHONY: run test install test-scenarios serve

# Keep tool caches/temp inside the repo so commands work in sandboxed environments.
UV_ENV = UV_CACHE_DIR=$(CURDIR)/.uv-cache TMPDIR=$(CURDIR)/.tmp XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.mplconfig

# Server report directory
serve:
	python -m http.server 8000 --directory output

# Install the package in editable mode
install:
	$(UV_ENV) uv pip install -e .

# Run the CLI with example arguments
run:
	$(UV_ENV) uv run dtm-differ run --a example.tif --b example.tif --out output/

# Test the CLI (you can customize this with your test command)
test:
	$(UV_ENV) uv run dtm-differ --help

# Generate test DTMs for edge case testing
generate-test-dtms:
	$(UV_ENV) uv run python -m dtm_differ.generate_test_dtms test_data/sample_dtms

# Run tests with generated scenarios (generates DTMs first if needed)
test-scenarios: generate-test-dtms
	$(UV_ENV) uv run pytest src/tests/test_pipeline_with_scenarios.py -v
