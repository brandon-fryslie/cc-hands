#!/bin/sh
# verify-wheel.sh WHEEL VERSION: install the wheel as a stranger would, with no Go and a fresh uv tool directory, and
# check that the hands it gives prints VERSION and the fritter inside its package runs a program.
set -eu
wheel=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
expected=$2
uv=$(command -v uv)
fresh=$(mktemp -d)
trap 'rm -rf "$fresh"' EXIT
# No go on this PATH: the install and everything it runs must do without one.
# Of this machine's environment, only how it reaches the network passes: its certificates and its proxy.
network=$(env | grep -E '^(SSL_CERT_FILE|SSL_CERT_DIR|HTTPS?_PROXY|https?_proxy|NO_PROXY|no_proxy)=' || [ $? -eq 1 ])
stranger() { env -i $network HOME="$fresh/home" PATH="$fresh/bin:/usr/bin:/bin" UV_TOOL_DIR="$fresh/tools" UV_TOOL_BIN_DIR="$fresh/bin" UV_CACHE_DIR="$fresh/cache" UV_PYTHON_INSTALL_DIR="$fresh/python" "$@"; }
mkdir -p "$fresh/home"
stranger sh -c '! command -v go' >/dev/null || { echo "verify-wheel: go is still on the stranger's PATH" >&2; exit 1; }
# A Mac has its own Python, 3.9, which uv would otherwise take; hands needs 3.12, which uv fetches.
stranger "$uv" tool install --quiet --python 3.12 "$wheel"
said=$(stranger hands --version)
[ "$said" = "hands $expected" ] || { echo "verify-wheel: hands --version said '$said', not 'hands $expected'" >&2; exit 1; }
fritter=$(stranger "$fresh/tools/hands/bin/python" -c 'import hands, pathlib; print(pathlib.Path(hands.__file__).parent / "bin" / "fritter")')
# fritter puts its own terminal into raw mode, so it runs under script's pseudo-terminal. The program checks that
# fritter published its socket, and exits 7 so that fritter is seen handing the program's own exit code back.
set +e
stranger script -q /dev/null "$fritter" -- /bin/sh -c 'test -S "$FRITTER_SOCKET" || exit 1; exit 7' </dev/null >/dev/null
code=$?
set -e
[ "$code" = 7 ] || { echo "verify-wheel: the packaged fritter at $fritter exited $code, not the program's 7" >&2; exit 1; }
echo "verify-wheel: $(basename "$wheel") installs without go, prints hands $expected, and its fritter runs"
