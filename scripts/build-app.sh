#!/bin/sh
# build-app.sh OUT: build hands.app into OUT/hands.app, signed with the Developer ID Application identity in this
# Mac's keychain under the hardened runtime, as notarization requires. Its version is the hands version this checkout
# builds, so the app and the wheel built from one commit say the same.
set -eu
[ $# -eq 1 ] || { echo "build-app: usage: build-app.sh OUT (given: $*)" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
app=$1/hands.app
# [LAW:one-source-of-truth] hatch-vcs's version, read from the checkout's own install, as `hands --version` says it.
version=$(cd "$root" && uv run --quiet python -c 'from importlib.metadata import version; print(version("hands"))')
minimum=$(plutil -extract LSMinimumSystemVersion raw "$root/app/Info.plist")
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp "$root/app/Info.plist" "$app/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string "$version" "$app/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$version" "$app/Contents/Info.plist"
swiftc -O -target "arm64-apple-macos$minimum" -o "$app/Contents/MacOS/hands" "$root/app/main.swift"
# codesign takes the identity by its name's start, and refuses when that names more than one.
codesign --force --options runtime --timestamp --entitlements "$root/app/hands.entitlements" --sign "Developer ID Application" "$app"
codesign --verify --strict --verbose=2 "$app"
echo "$app"
