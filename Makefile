.PHONY: check app notarized-app

check:
	uv run pytest
	uv run pyright
	cd fritter && go test -race ./...
	swiftc -typecheck app/main.swift

app:
	scripts/build-app.sh build

notarized-app: app
	scripts/notarize-app.sh build/hands.app
