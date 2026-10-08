#!/bin/sh
# build-app.sh OUT: build hands.app into OUT/hands.app, signed with the Developer ID Application identity in this
# Mac's keychain under the hardened runtime, as notarization requires. Its version is the release of the hands version
# this checkout builds.
set -eu
[ $# -eq 1 ] || { echo "build-app: usage: build-app.sh OUT (given: $*)" >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd)
app=$1/hands.app
# [LAW:one-source-of-truth] hatch-vcs's version, read from the checkout's own install, as `hands --version` says it;
# a bundle's version is release numbers alone, so a dev build between tags carries the release it leads to.
version=$(cd "$root" && uv run --quiet python -c 'from importlib.metadata import version; from packaging.version import Version; print(Version(version("hands")).base_version)')
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp "$root/app/Info.plist" "$app/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string "$version" "$app/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$version" "$app/Contents/Info.plist"
"$root/scripts/swiftc-app.sh" -O -o "$app/Contents/MacOS/hands"
# codesign takes the identity by its name's start, and refuses when that names more than one.
codesign --force --options runtime --timestamp --entitlements "$root/app/hands.entitlements" --sign "Developer ID Application" "$app"
codesign --verify --strict --verbose=2 "$app"
echo "$app"
