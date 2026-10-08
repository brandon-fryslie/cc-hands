"""`hands grant` against a stand-in macOS on a virtual clock: which app the grant goes to, what is said before the wait,
how the wait ends, and the event each run emits."""

from dataclasses import dataclass, field

import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice import grant
from hands.voice.grant import App, NoApp

TERMINAL = "/System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal"


@dataclass
class StandIn:
    """macOS as `hands grant` sees it: the grant is given once `given_after` seconds of the wait have gone by, if ever."""

    responsible: str = TERMINAL
    given_after: float | None = None
    clock: float = 0.0
    asked: list[str] = field(default_factory=list[str])

    def mac(self) -> grant.Mac:
        return grant.Mac(
            responsible=lambda: self.responsible,
            granted=lambda: self.given_after is not None and self.clock >= self.given_after,
            ask=lambda: self.asked.append("ask"),
            show=lambda: self.asked.append("show"),
            sleep=self.sleep,
            now=lambda: self.clock,
        )

    def sleep(self, seconds: float) -> None:
        self.clock += seconds


def given(stand_in: StandIn) -> tuple[int, WideEvent]:
    entries: list[Entry] = []
    code = grant.give(stand_in.mac(), entries.append)
    [event] = entries
    assert isinstance(event, WideEvent)
    return code, event


@pytest.mark.parametrize(
    ("executable", "grantee"),
    [
        (TERMINAL, App("Terminal")),
        ("/Applications/iTerm.app/Contents/MacOS/iTerm2", App("iTerm")),
        ("/Applications/hands.app/Contents/MacOS/hands", App("hands")),
        # A helper app inside another is its own app to TCC.
        ("/Applications/Outer.app/Contents/Helpers/Inner.app/Contents/MacOS/Inner", App("Inner")),
        ("/usr/libexec/sshd-session", NoApp("/usr/libexec/sshd-session")),
    ],
)
def test_the_grant_goes_to_the_innermost_app_around_the_responsible_process(executable: str, grantee: grant.Grantee) -> None:
    assert grant.grantee(executable) == grantee


def test_a_grant_already_given_asks_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    stand_in = StandIn(given_after=0)
    code, event = given(stand_in)
    assert code == 0 and stand_in.asked == [] and stand_in.clock == 0
    assert "Terminal, the app hands runs in, has the Input Monitoring grant" in capsys.readouterr().out
    assert event.outcome == "ok" and event.facts["before"] is True and event.facts["grantee"] == App("Terminal")


def test_a_grant_given_during_the_wait_names_the_app_and_the_pane_and_says_to_choose_later(capsys: pytest.CaptureFixture[str]) -> None:
    stand_in = StandIn(given_after=30)
    code, event = given(stand_in)
    said = capsys.readouterr().out
    assert code == 0 and stand_in.asked == ["ask", "show"]
    assert "Privacy & Security > Input Monitoring: turn Terminal on" in said and "choose Later" in said
    assert said.index("choose Later") < said.index("Terminal has the Input Monitoring grant")
    assert event.outcome == "ok" and (event.facts["before"], event.facts["after"], event.facts["polls"]) == (False, True, 30)


def test_a_grant_not_given_within_the_wait_exits_1_saying_a_second_run_waits_again(capsys: pytest.CaptureFixture[str]) -> None:
    stand_in = StandIn(given_after=None)
    code, event = given(stand_in)
    assert code == 1 and stand_in.clock == grant.WAIT_SECONDS
    assert "within 5 minutes; running this again waits again" in capsys.readouterr().err
    assert event.outcome == "failed" and (event.facts["after"], event.facts["polls"]) == (False, grant.WAIT_SECONDS)


def test_over_ssh_no_app_holds_hands_so_it_exits_2_at_once_asking_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    stand_in = StandIn(responsible="/usr/libexec/sshd-session")
    code, event = given(stand_in)
    assert code == 2 and stand_in.asked == [] and stand_in.clock == 0
    assert "/usr/libexec/sshd-session, which is no app" in capsys.readouterr().err
    assert event.outcome == "failed" and event.facts["grantee"] == NoApp("/usr/libexec/sshd-session")
