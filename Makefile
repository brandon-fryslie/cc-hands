.PHONY: check

# GOTOOLCHAIN=local: a go.mod that outgrows the installed Go fails here instead of downloading a second toolchain.
check:
	uv run pytest
	uv run pyright
	cd fritter && GOTOOLCHAIN=local go test -race ./...
