"""Giving back what a session's tap replaced: the one rule, read the same in Python and in sh."""

import subprocess

import pytest

from hands.sessions.untap import REPLACED, tapped, untap_script, untapped

TAP = "http://127.0.0.1:40000"
# A session's environment as fritter left it, its proxy and trust both replaced, and what they were before kept.
SESSION = {
    "HOME": "/home/a",
    "FRITTER_SOCKET": "/tmp/fritter-1/session.sock",
    "FRITTER_TAP": TAP,
    "HTTPS_PROXY": TAP,
    "https_proxy": TAP,
    "HTTP_PROXY": TAP,
    "http_proxy": TAP,
    "NODE_EXTRA_CA_CERTS": "/tmp/fritter-1/trusted.pem",
    "FRITTER_OUTER_HTTPS_PROXY": "http://corp:3128",
    "FRITTER_OUTER_NODE_EXTRA_CA_CERTS": "",
}

CASES = {
    "tapped": SESSION,
    "tapped with nothing replaced that was set": {name: value for name, value in SESSION.items() if not name.startswith("FRITTER_OUTER_")},
    "a proxy set again since": {**SESSION, "HTTPS_PROXY": "http://other:8080"},
    "not a session's": {"HOME": "/home/a", "HTTPS_PROXY": "http://corp:3128"},
}


def test_a_tapped_session_s_environment_is_given_back_what_was_there_before() -> None:
    assert untapped(SESSION) == {
        "HOME": "/home/a",
        # Not the tap's: the session's own, which a claude started from it runs under a fritter of its own.
        "FRITTER_SOCKET": "/tmp/fritter-1/session.sock",
        "HTTPS_PROXY": "http://corp:3128",
        # Set and empty before is set and empty again, not unset.
        "NODE_EXTRA_CA_CERTS": "",
    }
    assert tapped(SESSION) and not tapped(untapped(SESSION))


@pytest.mark.parametrize("environ", CASES.values(), ids=CASES.keys())
def test_sh_gives_back_what_python_does(environ: dict[str, str]) -> None:
    printed = subprocess.run(["/bin/sh", "-c", untap_script() + "env -0"], env=environ, capture_output=True, text=True, check=True).stdout
    after = dict(entry.split("=", 1) for entry in printed.split("\0") if entry)
    watched = {*environ, *REPLACED}
    assert {name: value for name, value in after.items() if name in watched} == untapped(environ)
