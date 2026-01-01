.PHONY: run test install

# Keep tool caches/temp inside the repo so commands work in sandboxed environments.
UV_ENV = UV_CACHE_DIR=$(CURDIR)/.uv-cache TMPDIR=$(CURDIR)/.tmp XDG_CACHE_HOME=$(CURDIR)/.cache MPLCONFIGDIR=$(CURDIR)/.mplconfig

# Install the package in editable mode
install:
	$(UV_ENV) uv pip install -e .

# Run the CLI with example arguments
run:
	$(UV_ENV) uv run dtm-differ run --a example.tif --b example.tif --out output/

# Test the CLI (you can customize this with your test command)
test:
	$(UV_ENV) uv run dtm-differ --help
