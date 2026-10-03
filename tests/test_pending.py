"""Pending speech: the order what hands has to tell is told in, and what folds or drops as the floor lets it go."""

from collections.abc import Mapping, Sequence

import pytest

from hands.core.effects import Asking, DeadlineNear, Expired, ModeChanged, Narrate, Note, SessionGone, Speak
from hands.core.narration import THE_TESTS, Segment
from hands.core.pending import Briefing, Finished, News, Pending, Unread, coalesce
from hands.core.session import Held, Permission, PromptId, RequestId, SessionId

API, WEB = SessionId("api"), SessionId("web")
BASH = Permission("Bash", {"command": "ls"})
EDIT = Permission("Edit", {"file_path": "a.py"})


def news(reply: str) -> News:
    return News(PromptId(reply), reply, "", "", (Segment(THE_TESTS, "one test run"),), frozenset())


def finished(session: SessionId, *replies: str) -> Finished:
    return Finished(session, tuple(news(reply) for reply in replies), "full")


def asks(session: SessionId, request: str, on: Permission = BASH) -> Narrate:
    return Narrate(Asking(session, RequestId(request), on))


def held(session: SessionId, request: str, on: Permission = BASH) -> Mapping[SessionId, Held]:
    return {session: Held(on, RequestId(request), deadline=60.0, warned=False)}


NOTE = Note(ModeChanged(API, "acceptEdits"))
GONE = SessionGone(SessionId("old"))


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
            (finished(API, "one"), finished(WEB, "two"), asks(API, "a2")),
            held(API, "a2"),
            (finished(API, "one"), asks(API, "a2"), finished(WEB, "two")),
            id="a session's story is told in the order it happened, its turn before its next request",
        ),
        pytest.param(
            (finished(API, "one"), Unread(API), finished(API, "three")),
            {},
            (finished(API, "one"), Unread(API), finished(API, "three")),
            id="a turn that could not be read stands between the turns it came between",
        ),
        pytest.param(
            (finished(API, "one"), NOTE, finished(API, "two")),
            {},
            (NOTE, finished(API, "one", "two")),
            id="what is only known neither breaks a fold nor waits behind it",
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
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)), Speak(Expired(API, BASH))),
            {},
            (Speak(Expired(API, BASH)),),
            id="a deadline counted down on a dialog that expired is not told, its expiry is",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)),),
            held(API, "a2", EDIT),
            (),
            id="a deadline on a dialog another replaced is not told",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)),),
            held(API, "a1"),
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)),),
            id="a deadline on the dialog still waiting is told",
        ),
        pytest.param(
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)),),
            held(API, "a2"),
            (),
            id="a deadline on a request asked again the same way is not told",
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
    first, second = News(None, "one", "", "", (Segment(THE_TESTS, "one test run"),), frozenset()), News(None, "two", "It committed.", "Push it?", (), frozenset())
    [folded] = coalesce((Finished(API, (first,), "full"), Finished(API, (second,), "brief")), {})
    assert folded == Finished(API, (first, second), "brief")
