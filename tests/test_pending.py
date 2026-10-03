"""Pending speech: the order what hands has to tell is told in, and what folds or drops as the floor lets it go."""

from collections.abc import Mapping, Sequence

import pytest

from hands.core.effects import Asking, DeadlineNear, Expired, ModeChanged, Narrate, Note, SessionGone, Speak
from hands.core.narration import THE_TESTS, Segment
from hands.core.pending import Briefing, Finished, News, Pending, Unread, coalesce
from hands.core.session import Held, Permission, RequestId, SessionId

API, WEB = SessionId("api"), SessionId("web")
BASH = Permission("Bash", {"command": "ls"})
EDIT = Permission("Edit", {"file_path": "a.py"})


def news(reply: str) -> News:
    return News(reply, "", "", (Segment(THE_TESTS, "one test run"),))


def finished(session: SessionId, *replies: str) -> Finished:
    return Finished(session, tuple(news(reply) for reply in replies))


def asks(session: SessionId, request: str, on: Permission = BASH) -> Narrate:
    return Narrate(Asking(session, RequestId(request), on))


def held(session: SessionId, request: str, on: Permission = BASH) -> Mapping[SessionId, Held]:
    return {session: Held(on, RequestId(request), deadline=60.0, warned=False)}


NOTE = Note(ModeChanged(API, "acceptEdits"))
GONE = SessionGone(WEB)


@pytest.mark.parametrize(
    ("pending", "waiting", "told"),
    [
        pytest.param((), {}, (), id="nothing pending"),
        pytest.param(
            (finished(API, "one"), GONE, NOTE, asks(WEB, "w1")),
            held(WEB, "w1"),
            (NOTE, asks(WEB, "w1"), finished(API, "one"), GONE),
            id="known, then blocking, then result, then fyi",
        ),
        pytest.param(
            (finished(API, "one"), finished(WEB, "two"), Unread(API)),
            {},
            (finished(API, "one"), finished(WEB, "two"), Unread(API)),
            id="arrival order within a priority",
        ),
        pytest.param(
            (finished(API, "one"), finished(WEB, "two"), finished(API, "three")),
            {},
            (finished(API, "one", "three"), finished(WEB, "two")),
            id="a session's turns fold into one telling where its first stood",
        ),
        pytest.param(
            (finished(API, "one"), finished(API, "two"), SessionGone(API)),
            {},
            (finished(API, "one", "two"), SessionGone(API)),
            id="a session gone is told after the turns it ended",
        ),
        pytest.param(
            (asks(API, "a1"), asks(API, "a2")),
            held(API, "a2"),
            (asks(API, "a2"),),
            id="a request answered at the keyboard while held is not told",
        ),
        pytest.param(
            (asks(API, "a1"),),
            {},
            (),
            id="a request nothing waits on any more is not told",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, BASH, 10.0)), Speak(Expired(API, BASH))),
            {},
            (Speak(Expired(API, BASH)),),
            id="a deadline counted down on a dialog that expired is not told, its expiry is",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, BASH, 10.0)),),
            held(API, "a2", EDIT),
            (),
            id="a deadline on a dialog another replaced is not told",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, BASH, 10.0)),),
            held(API, "a1"),
            (Speak(DeadlineNear(API, BASH, 10.0)),),
            id="a deadline on the dialog still waiting is told",
        ),
        pytest.param(
            (asks(API, "a1"), Briefing("how the sessions stood")),
            held(API, "a1"),
            (Briefing("how the sessions stood"), asks(API, "a1")),
            id="the briefing is known before anything is said",
        ),
    ],
)
def test_coalesce(pending: Sequence[Pending], waiting: Mapping[SessionId, Held], told: tuple[Pending, ...]) -> None:
    assert coalesce(pending, waiting) == told


def test_a_folded_telling_keeps_every_turn_s_parts() -> None:
    first, second = News("one", "", "", (Segment(THE_TESTS, "one test run"),)), News("two", "It committed.", "Push it?", ())
    [folded] = coalesce((Finished(API, (first,)), Finished(API, (second,))), {})
    assert folded == Finished(API, (first, second))
