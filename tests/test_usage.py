"""The brain's token usage, read off the wire's reply frames, and the tool that reports it."""

from typing import cast

from hands.brain.usage import Spent, Tally, Usage
from hands.core.session import SessionId
from hands.core.wire import Compaction, Exchanged, Heard, Kind, MainTurn, MessageDelta, MessageStarted, Message, Reached, Sent, Streamed
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, root
from hands.voice.tools import Called, audited, usage_tool

BRAIN = SessionId("brain")


def sent(exchange: str, kind: Kind = MainTurn(None), session: SessionId = BRAIN) -> Sent:
    return Sent(exchange, session, kind, {})


def started(exchange: str, input: int, cached: int = 0, created: int = 0, model: str = "claude-opus-5-5") -> Heard:
    usage = {"input_tokens": input, "cache_read_input_tokens": cached, "cache_creation_input_tokens": created, "output_tokens": 1}
    return Heard(exchange, MessageStarted(f"msg_{exchange}", model, usage))


def delta(exchange: str, output: int) -> Heard:
    return Heard(exchange, MessageDelta("end_turn", {"output_tokens": output}))


def whole(exchange: str, kind: Kind = MainTurn(None)) -> Exchanged:
    message = Message("m", "claude-opus-5-5", (), "end_turn", {})
    return Exchanged(exchange, BRAIN, kind, "POST", "/v1/messages", 1, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 1, Streamed(message)), False, root())


def heard(usage: Usage, *observed: Sent | Heard | Exchanged) -> Usage:
    for each in observed:
        usage.hear(each)
    return usage


def test_nothing_is_counted_before_a_main_turn_reply_begins() -> None:
    assert heard(Usage(BRAIN), sent("a")).reading() is None


def test_the_context_is_the_latest_main_turn_request_with_its_reply_however_it_was_cached() -> None:
    usage = heard(Usage(BRAIN), sent("a"), started("a", 2, cached=14800, created=98), delta("a", 40))
    assert usage.reading() == Tally("claude-opus-5-5", 2 + 14800 + 98 + 40, Spent(2 + 14800 + 98, 40, 1))


def test_a_reply_still_streaming_is_counted_so_a_tool_called_in_it_reads_its_own_request() -> None:
    # The tool runs once the reply ends in its call; its Exchanged comes after the last byte, so it may not be whole yet.
    usage = heard(Usage(BRAIN), sent("a"), started("a", 10, cached=1000), delta("a", 5), whole("a"), sent("b"), started("b", 3, cached=1015))
    reading = cast(Tally, usage.reading())
    assert reading.in_context == 3 + 1015 + 1
    assert reading.spent == Spent(10 + 1000 + 3 + 1015, 5 + 1, 2)


def test_a_compaction_is_spent_but_is_not_the_conversation_and_the_turn_after_it_is() -> None:
    usage = heard(
        Usage(BRAIN),
        sent("a"), started("a", 1, cached=90000), delta("a", 10), whole("a"),
        sent("c", Compaction()), started("c", 1, cached=90010), delta("c", 2000), whole("c", Compaction()),
    )
    assert cast(Tally, usage.reading()).in_context == 1 + 90000 + 10
    heard(usage, sent("b"), started("b", 2500), delta("b", 20))
    reading = cast(Tally, usage.reading())
    assert reading.in_context == 2500 + 20
    assert reading.spent == Spent(90001 + 90011 + 2500, 10 + 2000 + 20, 3)


def test_a_count_a_delta_gives_as_null_keeps_the_one_before() -> None:
    nulled = Heard("a", MessageDelta("end_turn", {"output_tokens": 40, "cache_read_input_tokens": None}))
    usage = heard(Usage(BRAIN), sent("a"), started("a", 2, cached=90000), nulled)
    assert cast(Tally, usage.reading()).in_context == 2 + 90000 + 40


def test_a_late_frame_of_an_earlier_main_turn_does_not_stand_for_the_conversation() -> None:
    usage = heard(Usage(BRAIN), sent("a"), started("a", 100), sent("b"), started("b", 200), delta("a", 7))
    assert cast(Tally, usage.reading()).in_context == 200 + 1


def test_another_sessions_replies_are_not_the_brains() -> None:
    other = SessionId("someone")
    usage = heard(Usage(BRAIN), sent("x", session=other), started("x", 5000), sent("a"), started("a", 10))
    assert usage.reading() == Tally("claude-opus-5-5", 11, Spent(10, 1, 1))


async def test_each_call_is_one_event_holding_what_it_reported() -> None:
    recorded: list[Entry] = []
    usage = Usage(BRAIN)
    context_usage = audited(usage_tool(usage), recorded.append)
    assert await context_usage.body() == {"error": "No reply of yours has been counted yet."}
    heard(usage, sent("a"), started("a", 2, cached=14800, created=98), delta("a", 40))
    reported = {"model": "claude-opus-5-5", "in_context_tokens": 14940, "spent": {"input_tokens": 14900, "output_tokens": 40, "replies": 1}}
    assert await context_usage.body() == reported
    events = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert [cast(Called, event.facts["called"]).result for event in events] == [{"error": "No reply of yours has been counted yet."}, reported]
