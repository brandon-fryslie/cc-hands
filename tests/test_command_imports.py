"""Which `hands` commands load Pipecat: status, check, and log answer without it."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Run in a fresh interpreter, since this one has Pipecat loaded by other tests: any import that reaches it, at the
# CLI's top or inside the command, fails here, whichever module it runs through. The modules are written to a file,
# so whatever the command prints is the command's own. `log` follows until Ctrl-C, which the timer stands in for.
CHILD = """
import json, signal, sys
from hands.daemon.cli import main

home, command, out = sys.argv[1:]
if command == "log":
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    signal.signal(signal.SIGALRM, signal.default_int_handler)
main(["--home", home, command])
with open(out, "w") as file:
    json.dump(sorted(m for m in sys.modules if m.split(".")[0] == "pipecat"), file)
"""


@pytest.mark.parametrize("command", ["status", "check", "log"])
def test_the_command_answers_without_loading_pipecat(command: str, tmp_path: Path) -> None:
    out = tmp_path / "modules.json"
    child = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path / "home"), command, str(out)], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(out.read_text()) == []
