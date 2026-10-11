"""`hands install-plugin`, as install.sh runs it at a terminal: Claude Code's own accept prompt is said before it is asked
and answered by the person alone, and a plugin already installed is not asked for again.

Claude Code is the fake built from recordings (claudecode_fake), with a person at its terminal who accepts or declines.
"""

import json
import os
from pathlib import Path

import pytest

from claudecode_fake import INSTALL, Config, Fake, Person, config_dir
from hands.daemon import cli
from hands.sessions.audit import segment
from hands.sessions.home import Home
from hands.sessions.hookconfig import MARKETPLACE, PLUGIN_ID

ANNOUNCED = f"told: Claude Code now shows the command `hands plugin`, which installs {PLUGIN_ID}, and asks whether to run it: answer y"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Home:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    return Home(tmp_path / "home")


def install(home: Home, fake: Fake) -> int:
    return cli.install_plugin(fake, fake.told, cli.audit_log_of(home).record)


def person_config(fake: Fake) -> Config:
    return fake.configs.setdefault(config_dir(os.environ), Config())


def events(home: Home) -> list[dict[str, object]]:
    lines = (json.loads(line) for line in segment(home.audit, 0).read_text().splitlines())
    return [line for line in lines if line.get("event") == "plugin.install"]


def asked(fake: Fake) -> list[str]:
    return [line for line in fake.transcript if not line.startswith(("told: ", "shown: "))]


def test_the_accept_prompt_is_announced_before_claude_code_asks_it(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake()
    assert install(home, fake) == 0
    assert fake.transcript.index(ANNOUNCED) < fake.transcript.index(f"shown: {INSTALL['asked']}")
    assert f"the plugin {PLUGIN_ID} is installed and enabled for every session" in capsys.readouterr().out
    assert asked(fake) == ["plugin list", "plugin marketplace list", f"plugin marketplace add {MARKETPLACE}", f"plugin install {PLUGIN_ID}", "plugin list"]
    # [LAW:nothing-unseen] the install's event: what was found before, what the install exited with, what was there after.
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 0, "after": "ready"})


def test_declining_the_accept_prompt_fails_naming_the_plugin_as_not_installed(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(Person(plugin="declines"))
    assert install(home, fake) == 1
    assert f"hands install-plugin: the plugin {PLUGIN_ID} is not installed" in capsys.readouterr().err
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 0, "install_exit": 1, "after": "missing"})


def test_a_plugin_already_installed_is_not_asked_for_again(home: Home) -> None:
    fake = Fake()
    person_config(fake).plugin = True
    assert install(home, fake) == 0
    assert fake.transcript == ["plugin list"]
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "ready"})


def test_a_marketplace_that_cannot_be_added_fails_before_anything_is_asked(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(repository=False)
    assert install(home, fake) == 2
    assert f"`claude plugin marketplace add {MARKETPLACE}` failed (1)" in capsys.readouterr().err
    assert f"plugin install {PLUGIN_ID}" not in asked(fake)
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "missing", "marketplace": "missing", "marketplace_add_exit": 1})


def test_a_claude_that_cannot_list_its_plugins_installs_nothing(home: Home, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake(unrunnable={"plugin list"})
    assert install(home, fake) == 2
    assert "cannot ask `claude plugin list`" in capsys.readouterr().err
    assert fake.transcript == []
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("failed", {"before": "unknown"})


def test_a_marketplace_the_person_added_from_a_checkout_is_installed_from_as_it_is(home: Home) -> None:
    fake = Fake()
    person_config(fake).marketplace = "checkout"
    assert install(home, fake) == 0
    assert not [call for call in asked(fake) if call.startswith("plugin marketplace add")]
    [event] = events(home)
    assert (event["outcome"], event["facts"]) == ("ok", {"before": "missing", "marketplace": "ready", "install_exit": 0, "after": "ready"})
