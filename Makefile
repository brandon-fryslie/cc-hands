.PHONY: check

check:
	uv run pytest
	uv run pyright
	cd fritter && go test -race ./...
