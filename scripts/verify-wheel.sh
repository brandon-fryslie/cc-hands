#!/bin/sh
# verify-wheel.sh WHEEL CONSTRAINTS VERSION: install the wheel as a stranger would, within the locked versions
# CONSTRAINTS names, with no Go and a fresh uv tool directory, and check that the hands it gives prints VERSION, the fritter inside its package runs a program, and install-fritter puts
# that fritter and its claude in the stranger's hands home, and hands plugin renders a plugin whose launcher runs
# hands. The stranger has Homebrew's portaudio, which PyAudio builds against.
set -eu
[ $# -eq 3 ] || { echo "verify-wheel: usage: verify-wheel.sh WHEEL CONSTRAINTS VERSION (given: $*)" >&2; exit 2; }
wheel=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
constraints=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
expected=$3
uv=$(command -v uv)
fresh=$(mktemp -d)
trap 'rm -rf "$fresh"' EXIT
# No go on this PATH: the install and everything it runs must do without one.
# Of this machine's environment, only how it reaches the network passes: its certificates and its proxy.
# Each passes as one word, whatever spaces or globs its value holds.
stranger() {
  for name in SSL_CERT_FILE SSL_CERT_DIR HTTPS_PROXY HTTP_PROXY https_proxy http_proxy NO_PROXY no_proxy; do
    value=$(printenv "$name") && set -- "$name=$value" "$@"
  done
  env -i HOME="$fresh/home" PATH="$fresh/bin:/usr/bin:/bin" UV_TOOL_DIR="$fresh/tools" UV_TOOL_BIN_DIR="$fresh/bin" UV_CACHE_DIR="$fresh/cache" UV_PYTHON_INSTALL_DIR="$fresh/python" "$@"
}
mkdir -p "$fresh/home"
stranger sh -c '! command -v go' >/dev/null || { echo "verify-wheel: go is still on the stranger's PATH" >&2; exit 1; }
# A Mac has its own Python, 3.9, which uv would otherwise take; hands needs 3.12, which uv fetches.
stranger "$uv" tool install --quiet --python 3.12 --constraints "$constraints" "$wheel"
said=$(stranger hands --version)
[ "$said" = "hands $expected" ] || { echo "verify-wheel: hands --version said '$said', not 'hands $expected'" >&2; exit 1; }
# Where install-fritter copies from, as hands itself names it, so the check is of the path hands uses.
fritter=$(stranger "$fresh/tools/hands/bin/python" -c 'from hands.sessions.wrapper import PACKAGED; print(PACKAGED)')
# fritter puts its own terminal into raw mode, so it runs under script's pseudo-terminal. The program checks that
# fritter published its socket, and exits 7 so that fritter is seen handing the program's own exit code back.
set +e
stranger script -q /dev/null "$fritter" -- /bin/sh -c 'test -S "$FRITTER_SOCKET" || exit 1; exit 7' </dev/null >/dev/null
code=$?
set -e
[ "$code" = 7 ] || { echo "verify-wheel: the packaged fritter at $fritter exited $code, not the program's 7" >&2; exit 1; }
# The home's bin first on PATH, as the README has the stranger put it, so install-fritter finds its claude and exits 0.
# A later PATH= in env's arguments wins over stranger's own.
stranger PATH="$fresh/home/.hands/bin:$fresh/bin:/usr/bin:/bin" hands install-fritter >/dev/null
cmp -s "$fritter" "$fresh/home/.hands/bin/fritter" || { echo "verify-wheel: install-fritter did not copy the packaged fritter to the home's bin" >&2; exit 1; }
# The plugin the marketplace entry installs, as Claude Code gets it: hands plugin prints its directory, which holds the
# manifest and hooks, and its launcher runs the stranger's hands on the shim the hooks name.
plugin=$(stranger hands plugin)
for file in .claude-plugin/plugin.json hooks/hooks.json; do
  [ -f "$plugin/$file" ] || { echo "verify-wheel: the plugin hands plugin printed, $plugin, has no $file" >&2; exit 1; }
done
stranger "$plugin/hooks/python" -c 'import hands.sessions.shim' || { echo "verify-wheel: the plugin's launcher at $plugin/hooks/python cannot import the shim" >&2; exit 1; }
echo "verify-wheel: $(basename "$wheel") installs without go, prints hands $expected, its fritter runs, install-fritter installs it, and hands plugin renders a plugin whose launcher runs it"
