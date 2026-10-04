"""Which `hands` commands load Pipecat: status, check, log, phone, and login answer without it."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Run in a fresh interpreter, since this one has Pipecat loaded by other tests: any import that reaches it, at the
# CLI's top or inside the command, fails here, whichever module it runs through. The modules are written to a file,
# so whatever the command prints is the command's own. Each command's reach past this machine has a stand-in: `log`
# follows until Ctrl-C, which its first wait for a new line stands in for; `phone` asks Tailscale for the tailnet's
# name and the machine for its LAN addresses; `login` runs Claude Code's own login at the terminal.
CHILD = """
import json, sys, types
from hands.daemon import cli

def interrupted(seconds):
    raise KeyboardInterrupt

async def untailed():
    from hands.voice.phoneaddress import Untailed
    return Untailed("a stand-in for Tailscale")

def logged_in(config_dir, base_url, inherited):
    from hands.brain.process import Login
    return Login("someone@example.com", None)

home, command, out = sys.argv[1:]
if command == "log":
    cli.time = types.SimpleNamespace(sleep=interrupted)
if command == "phone":
    from hands.voice import phoneaddress
    phoneaddress.tailnet_name = untailed
    phoneaddress.lan_addresses = lambda: ["192.0.2.1"]
if command == "login":
    from hands.brain import process
    process.login = logged_in
code = cli.main(["--home", home, command])
with open(out, "w") as file:
    json.dump({"code": code, "pipecat": sorted(m for m in sys.modules if m.split(".")[0] == "pipecat")}, file)
"""


# Each command's code in a home with no daemon: status and check say it is not running, and the rest run whole.
@pytest.mark.parametrize(("command", "code"), [("status", 1), ("check", 1), ("log", 0), ("phone", 0), ("login", 0)])
def test_the_command_answers_without_loading_pipecat(command: str, code: int, tmp_path: Path) -> None:
    out = tmp_path / "modules.json"
    child = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path / "home"), command, str(out)], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(out.read_text()) == {"code": code, "pipecat": []}, child.stderr
