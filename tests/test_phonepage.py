"""What the phone's page learns from Tailscale's own command, and what becomes of that command when hands stops."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hands.sessions.home import Home
from hands.voice import phoneaddress
from hands.voice.phoneaddress import Tailnet, Untailed, tailnet, tailnet_name

NAMED = """#!/bin/sh
case "$1" in
status) echo '{"CertDomains": ["hands.example.ts.net"]}' ;;
*) %s ;;
esac
"""


def tailscale(directory: Path, cert: str) -> str:
    """A `tailscale` that says this machine's name, and runs `cert` when asked for its certificate; the PATH it is on."""
    return command(directory, NAMED % cert)


def command(directory: Path, script: str) -> str:
    """A `tailscale` that is `script`; the PATH it is on."""
    written = directory / "bin" / "tailscale"
    written.parent.mkdir()
    written.write_text(script)
    written.chmod(0o755)
    return f"{written.parent}{os.pathsep}{os.environ['PATH']}"


async def test_the_tailnet_is_the_name_tailscale_says_and_the_certificate_it_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", tailscale(tmp_path, "exit 0"))
    home = Home(tmp_path)
    assert await tailnet_name() == "hands.example.ts.net"
    assert await tailnet(home) == Tailnet("hands.example.ts.net", home.phone / "tailnet.crt", home.phone / "tailnet.key")


async def test_a_certificate_tailscale_refuses_is_untailed_in_its_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", tailscale(tmp_path, "echo 'not logged in' >&2; exit 3"))
    assert await tailnet(Home(tmp_path)) == Untailed("tailscale cert failed (3): not logged in")


async def test_a_tailscale_that_never_answers_is_untailed_and_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid = tmp_path / "pid"
    monkeypatch.setenv("PATH", command(tmp_path, f"#!/bin/sh\necho $$ > {pid}\nexec sleep 30\n"))
    # Long enough that the command has said its pid before it is killed, however loaded the machine.
    monkeypatch.setattr(phoneaddress, "TAILSCALE_TIMEOUT_SECONDS", 1.0)
    assert await tailnet_name() == Untailed("tailscale status took over 1s")
    # Gone, not merely dead: a killed child nobody reaped would still take the signal.
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


async def test_a_tailscale_that_cannot_be_started_is_untailed_in_the_systems_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No interpreter line: found on the PATH, and refused by the system when run.
    monkeypatch.setenv("PATH", command(tmp_path, "not a program\n"))
    match await tailnet_name():
        case Untailed(reason=reason):
            assert reason.startswith("cannot run tailscale status: [Errno 8]")
        case name:
            raise AssertionError(f"named {name}")


# A loop that ends while Tailscale is asked: asyncio.run cancels the asking wherever its command happens to be, from
# `tailscale status` being spawned to `tailscale cert` running. Run in a process of its own, since a loop that never
# closes would take the test run down with it.
SHUT_DOWN = """
import asyncio, os, sys, time
from pathlib import Path
from hands.sessions.home import Home
from hands.voice.phoneaddress import tailnet

home = Home(Path(sys.argv[1]))

async def ends(after: float) -> None:
    asking = asyncio.ensure_future(tailnet(home))
    await asyncio.sleep(after)
    raise RuntimeError(time.monotonic())

slowest = 0.0
for step in range(30):
    try:
        asyncio.run(ends(step * 0.002))
    except RuntimeError as ended:
        slowest = max(slowest, time.monotonic() - ended.args[0])
    try:
        os.waitpid(-1, os.WNOHANG)
        sys.exit(f"a tailscale was left unreaped when the loop ended {step * 2}ms into asking it")
    except ChildProcessError:
        pass
print(slowest)
"""


def test_a_loop_ended_while_tailscale_is_asked_kills_and_reaps_it_and_closes(tmp_path: Path) -> None:
    """The daemon's shutdown cancels the page's asking wherever it is, and must not then wait for ever on a tailscale
    it spawned.

    Python 3.12's asyncio subprocesses did: cancelled while starting, they waited for an exit nothing would deliver."""
    environment = {**os.environ, "PATH": tailscale(tmp_path, "exec sleep 30")}
    try:
        ran = subprocess.run((sys.executable, "-c", SHUT_DOWN, str(tmp_path)), capture_output=True, text=True, timeout=60, env=environment)
    except subprocess.TimeoutExpired:
        raise AssertionError("a loop ended while tailscale was asked never finished closing") from None
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert float(ran.stdout) < 1.0, f"the slowest loop took {ran.stdout.strip()}s to close"
