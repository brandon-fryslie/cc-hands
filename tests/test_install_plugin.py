"""`hands install-plugin`, run at a terminal as install.sh runs it: Claude Code's own accept prompt is said before it is
asked and answered by the person alone, and a plugin already installed is not asked for again.

`claude` is a stand-in that records each call, lists the plugin as its state file says, and reads its [y/N] answer from
the terminal as Claude Code does.
"""

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions.hookconfig import MARKETPLACE, PLUGIN_ID

CLAUDE = r"""#!/bin/bash
echo "claude $*" >>"$ROOT/calls"
case "$1 $2" in
  "plugin list")
    [ ! -e "$ROOT/list-fails" ] || { echo "cannot list" >&2; exit 3; }
    cat "$ROOT/listed" 2>/dev/null || echo "[]" ;;
  "plugin marketplace")
    case "$3" in
      list) cat "$ROOT/marketplaces" 2>/dev/null || echo "[]" ;;
      add)
        [ ! -e "$ROOT/add-fails" ] || exit 4
        printf '[{"name": "cc-hands", "source": "github", "repo": "%s"}]' "$4" >"$ROOT/marketplaces" ;;
    esac ;;
  "plugin install")
    printf 'Run this command now? [y/N] '
    read -r answer
    if [ "$answer" = y ]; then
      printf '[{"id": "%s", "scope": "user", "enabled": true}]' "$PLUGIN_ID" >"$ROOT/listed"
    else
      echo Aborted.
      exit 1
    fi ;;
esac
"""
ANNOUNCED = f"Claude Code now shows the command `hands plugin`, which installs {PLUGIN_ID}, and asks whether to run it: answer y"


@pytest.fixture
def root() -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="plug-", dir="/tmp")).resolve()
    stand_in = root / "bin" / "claude"
    stand_in.parent.mkdir()
    stand_in.write_text(CLAUDE)
    stand_in.chmod(0o755)
    yield root
    shutil.rmtree(root)


def at_a_terminal(root: Path, typed_ahead: bytes, answer: bytes) -> tuple[int, str]:
    """Run `hands install-plugin` on a terminal holding typed_ahead before it starts, answer the [y/N] with answer once
    it is asked, and return its exit and everything the terminal showed."""
    controller, terminal = os.openpty()
    os.write(controller, typed_ahead)
    environment = {**os.environ, "PATH": f"{root / 'bin'}:/usr/bin:/bin", "ROOT": str(root), "PLUGIN_ID": PLUGIN_ID, "HANDS_HOME": str(root / "home")}
    process = subprocess.Popen([sys.executable, "-m", "hands.daemon", "install-plugin"], env=environment, stdin=terminal, stdout=terminal, stderr=terminal, start_new_session=True)
    os.close(terminal)
    shown = b""
    answered = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready, _, _ = select.select([controller], [], [], 0.1)
        if not ready:
            if process.poll() is not None:
                break
            continue
        try:
            chunk = os.read(controller, 4096)
        except OSError:  # EIO: the last process holding the terminal let go of it
            break
        if not chunk:
            break
        shown += chunk
        if not answered and b"[y/N]" in shown:
            os.write(controller, answer)
            answered = True
    exit = process.wait(timeout=10)
    os.close(controller)
    return exit, shown.decode().replace("\r\n", "\n")


def events(root: Path) -> list[dict[str, object]]:
    lines = (json.loads(line) for line in segment(Home(root / "home").audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "plugin.install"]


def calls(root: Path) -> list[str]:
    return (root / "calls").read_text().splitlines()


def test_the_accept_prompt_is_announced_and_a_key_typed_before_it_is_not_taken_as_the_answer(root: Path) -> None:
    # An n pressed during the run's earlier steps sits in the terminal; the y the person gives at the prompt answers it.
    exit, shown = at_a_terminal(root, typed_ahead=b"n\n", answer=b"y\n")
    assert exit == 0, shown
    assert shown.index(ANNOUNCED) < shown.index("[y/N]")
    assert f"the plugin {PLUGIN_ID} is installed and enabled for every session" in shown
    list_call = "claude plugin list --json"
    added = [list_call, "claude plugin marketplace list --json", f"claude plugin marketplace add {MARKETPLACE}"]
    assert calls(root) == [*added, f"claude plugin install --scope user {PLUGIN_ID}", list_call]
    # [LAW:nothing-unseen] the install's event: what was found before, what the install exited with, what was there after.
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 0, "after": "ready"})


def test_declining_the_accept_prompt_fails_naming_the_plugin_as_not_installed(root: Path) -> None:
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"n\n")
    assert exit == 1
    assert f"hands install-plugin: the plugin {PLUGIN_ID} is not installed" in shown
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 1, "after": "missing"})


def test_a_plugin_already_installed_is_not_asked_for_again(root: Path) -> None:
    (root / "listed").write_text(json.dumps([{"id": PLUGIN_ID, "scope": "user", "enabled": True}]))
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 0, shown
    assert ANNOUNCED not in shown and "[y/N]" not in shown
    assert calls(root) == ["claude plugin list --json"]
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "ready"})


def test_a_marketplace_that_cannot_be_added_fails_before_anything_is_asked(root: Path) -> None:
    (root / "add-fails").touch()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 2
    assert f"`claude plugin marketplace add {MARKETPLACE}` failed (4)" in shown and "[y/N]" not in shown
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 4})


def test_a_claude_that_cannot_list_its_plugins_installs_nothing(root: Path) -> None:
    (root / "list-fails").touch()
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 2
    assert "`claude plugin list` failed (3)" in shown and "cannot list" in shown
    assert calls(root) == ["claude plugin list --json"]
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "unknown"})


def test_a_marketplace_the_person_added_from_a_checkout_is_installed_from_as_it_is(root: Path) -> None:
    (root / "marketplaces").write_text(json.dumps([{"name": "cc-hands", "source": "directory", "path": "/code/cc-hands"}]))
    exit, shown = at_a_terminal(root, typed_ahead=b"", answer=b"y\n")
    assert exit == 0, shown
    assert not [call for call in calls(root) if call.startswith("claude plugin marketplace add")]
    [event] = events(root)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "missing", "marketplace": "ready", "install_exit": 0, "after": "ready"})
