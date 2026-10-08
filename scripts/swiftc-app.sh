#!/bin/sh
# swiftc-app.sh ARGS...: swiftc on hands.app's launcher, app/main.swift, for the oldest macOS hands runs on, so a type
# check and a build agree on which APIs there are. ARGS say what to make of it: -typecheck, or -O -o OUT.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
# [LAW:one-source-of-truth] the oldest macOS is the one Info.plist declares.
minimum=$(plutil -extract LSMinimumSystemVersion raw "$root/app/Info.plist")
exec swiftc -target "arm64-apple-macos$minimum" "$@" "$root/app/main.swift"
