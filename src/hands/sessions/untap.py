"""What fritter's tap replaces in a session's environment, and how what the session runs is given it back.

A tapped session reaches its API with fritter as its proxy, trusting the authority fritter answers the API's host with
(fritter/tap.go). Both end with the session, so what is given back here is not left with them: a claude run inside it
taps itself or reaches the API directly, the session's shell commands run as they would outside it, and hands' brain is
the same brain wherever hands was started. Each is given back what the session's own environment held before fritter.
What Claude Code starts on its own - its MCP servers, its hooks - inherits the tap as it is.
"""

import shlex
from collections.abc import Mapping

# The variable Claude Code reads the certificates it trusts besides its own from.
TRUST = "NODE_EXTRA_CA_CERTS"

# [LAW:one-source-of-truth] what fritter sets in a tapped session: its address in each proxy variable (`proxied` in
# fritter/tap.go), each NO_PROXY empty (`exempted`), and in TRUST a file holding its authority. It keeps each one's
# earlier value in FRITTER_OUTER_<name>.
REPLACED = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy", TRUST)


def _outer(name: str) -> str:
    return f"FRITTER_OUTER_{name}"


def tapped(environ: Mapping[str, str]) -> bool:
    """Whether environ is a tapped session's, its proxy still the tap's: one set again since is what it was set to."""
    return bool(tap := environ.get("FRITTER_TAP")) and environ.get("HTTPS_PROXY") == tap


# What fritter keeps of a tap beside what it replaced: never left behind, tapped or not, so a tap's earlier values are
# never given back to a session they were not taken from.
KEPT = ("FRITTER_TAP", *map(_outer, REPLACED))


def untapped(environ: Mapping[str, str]) -> dict[str, str]:
    """environ with what its session's tap replaced given back, and the tap's own variables gone."""
    kept = {name: value for name, value in environ.items() if name not in KEPT}
    if not tapped(environ):
        return kept
    return {name: value for name, value in kept.items() if name not in REPLACED} | {name: environ[_outer(name)] for name in REPLACED if _outer(name) in environ}


def untap_script() -> str:
    """untapped, as sh run in the environment to give back."""
    given = "".join(
        f'  if [ -n "${{{_outer(name)}+x}}" ]; then export {name}="${_outer(name)}"; else unset {name}; fi\n' for name in REPLACED
    )
    return f'if [ -n "${{FRITTER_TAP-}}" ] && [ "${{HTTPS_PROXY-}}" = "$FRITTER_TAP" ]; then\n{given}fi\nunset {shlex.join(KEPT)}\n'
