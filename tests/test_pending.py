"""Pending speech: the order what hands has to tell is told in, and what folds or drops as the floor lets it go."""

from collections.abc import Mapping, Sequence

from pathlib import Path

import pytest

from hands.core.effects import Asking, DeadlineNear, Expired, Narrate, SessionGone, Speak
from hands.core.narration import THE_TESTS, Segment
from hands.core.pending import Coalesced, Finished, News, Pending, Unread, coalesce
from hands.core.session import Held, Membership, Permission, PromptId, RequestId, Running, Session, SessionId
from hands.core.status import Busy, Stamp

API, WEB = SessionId("api"), SessionId("web")
BASH = Permission("Bash", {"command": "ls"})
EDIT = Permission("Edit", {"file_path": "a.py"})


def news(reply: str) -> News:
    return News(PromptId(reply), reply, "", "", (Segment(THE_TESTS, "one test run"),), frozenset())


def finished(session: SessionId, *replies: str) -> Finished:
    return Finished(session, tuple(news(reply) for reply in replies), "full")


def asks(session: SessionId, request: str, on: Permission = BASH) -> Narrate:
    return Narrate(Asking(session, RequestId(request), on))


def held(session: SessionId, request: str, on: Permission = BASH) -> Mapping[SessionId, Session]:
    """The session live, its dialog waiting on an answer to `request`."""
    member = Membership(session, pid=4242, cwd=Path("/code/a"), transcript=Path("/code/a/t.jsonl"))
    return {session: Session(member, Running(Busy(), Stamp(1000), None), mode=None, dialog=Held(on, RequestId(request), deadline=60.0, warned=False, expiry="hook"))}


GONE = SessionGone(SessionId("old"))


@pytest.mark.parametrize(
    ("pending", "waiting", "told"),
    [
        pytest.param((), {}, (), id="nothing pending"),
        pytest.param(
            (finished(API, "one"), GONE, asks(WEB, "w1")),
            held(WEB, "w1"),
            (asks(WEB, "w1"), finished(API, "one"), GONE),
            id="blocking, then result, then fyi",
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
            (Speak(DeadlineNear(API, RequestId("a1"), BASH, 10.0)), Speak(Expired(API, BASH, "hook"))),
            {},
            (Speak(Expired(API, BASH, "hook")),),
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
    ],
)
def test_coalesce(pending: Sequence[Pending], waiting: Mapping[SessionId, Session], told: tuple[Pending, ...]) -> None:
    assert tuple(each.pending for each in coalesce(pending, waiting)) == told


def test_what_is_told_names_where_each_thing_it_tells_stood_and_what_was_dropped_is_named_by_none() -> None:
    """The floor follows each thing it held to its fate by these: a fold tells every turn it took in, and a request
    answered meanwhile is told by nothing."""
    first, second = News(None, "one", "", "", (), frozenset()), News(None, "two", "", "", (), frozenset())
    told = coalesce((Finished(API, (first,), "full"), asks(API, "gone"), GONE, Finished(API, (second,), "full")), {})
    assert told == (Coalesced(Finished(API, (first, second), "full"), (0, 3)), Coalesced(GONE, (2,)))


def test_a_folded_telling_keeps_every_turn_s_parts() -> None:
    first, second = News(None, "one", "", "", (Segment(THE_TESTS, "one test run"),), frozenset()), News(None, "two", "It committed.", "Push it?", (), frozenset())
    [folded] = coalesce((Finished(API, (first,), "full"), Finished(API, (second,), "brief")), {})
    assert folded.pending == Finished(API, (first, second), "brief")
