"""The brain's context kept from the proxy: old results as a line, in batches, from sentences forks said while the results were whole."""

import asyncio
from collections.abc import Callable, Mapping
from typing import cast

import pytest

from hands.brain.context import VOICE_COMPACTION, Keeper, Kept
from hands.brain.process import ForkFailed
from hands.core.context import LONG, Result, aged, boundary, key, line, question, sentence
from hands.core.sentences import Digest
from hands.core.session import SessionId
from hands.core.wire import (
    COMPACTION_INSTRUCTIONS,
    COMPACTION_OPENING,
    COMPACTION_REMINDER,
    SIDE_QUESTION_OPENING,
    Compaction,
    CountTokens,
    Exchanged,
    Fork,
    Hold,
    Kind,
    MainTurn,
    Message,
    Reached,
    Route,
    Send,
    Sent,
    Steer,
    Streamed,
    Stub,
    Tail,
    Text,
    Observed,
    ToolUse,
    classify,
    edited,
)
from hands.sessions.audit import Entry, ResultsStubbed

BRAIN = SessionId("brain")
MARKED = {"type": "ephemeral"}


def prompt(text: str, marked: bool = False) -> dict[str, object]:
    block: dict[str, object] = {"type": "text", "text": text}
    return {"role": "user", "content": [{**block, "cache_control": MARKED} if marked else block]}


def turn(n: int, text: str | None = None) -> list[dict[str, object]]:
    """One finished turn of the brain's: asked, a read, its result, and the reply."""
    call = f"call{n}"
    return [
        prompt(f"question {n}"),
        {"role": "assistant", "content": [{"type": "tool_use", "id": call, "name": "Read", "input": {"file_path": f"/{n}"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call, "content": [{"type": "text", "text": text or f"{n}" * LONG}]}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {n}"}]},
    ]


def history(finished: int, short: frozenset[int] = frozenset()) -> dict[str, object]:
    """The brain's main turn after `finished` turns: their history, and the prompt of the one now asked."""
    messages = [message for n in range(finished) for message in turn(n, "ok" if n in short else None)]
    return {"model": "claude-opus-5-5", "tools": [{"name": "Read"}], "messages": [*messages, prompt(f"question {finished}", marked=True)]}


def test_the_boundary_moves_once_every_k_turns_and_every_result_before_it_is_more_than_k_turns_old() -> None:
    assert [boundary(held, 3) for held in range(0, 14)] == [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9]
    # The newest result stubbed is from the turn before the boundary, more than K turns before the one in flight.
    assert all((held - 1) - (boundary(held, 3) - 1) > 3 for held in range(1, 40) if boundary(held, 3))


def test_only_long_results_from_before_the_boundary_are_aged() -> None:
    body = history(6, short=frozenset({1}))
    assert [result.call for result in aged(body, 3)] == ["call0", "call2"]
    assert aged(history(5), 3) == ()


def edited_by_rule(body: dict[str, object], every: int) -> list[object]:
    """The history as the proxy sends it when every aged result has a sentence."""
    stubs = tuple(Stub(result.call, line(result, f"file {result.turn} holds its digit")) for result in aged(body, every))
    return cast(list[object], edited(body, stubs)["messages"])


def stubbed(messages: list[object]) -> list[str]:
    blocks = [block for message in messages for block in message["content"]]  # pyright: ignore
    return [block["tool_use_id"] for block in blocks if block["type"] == "tool_result" and isinstance(block["content"], str)]  # pyright: ignore


def test_the_trim_stubs_exactly_the_old_results_and_the_prefix_is_byte_identical_between_batches() -> None:
    every = 3
    sent = [edited_by_rule(history(finished), every) for finished in range(0, 16)]
    assert [stubbed(messages) for messages in sent] == [[f"call{n}" for n in range(boundary(finished + 1, every))] for finished in range(0, 16)]
    # The history before each new prompt is what the last request sent, except at the turns a batch is stubbed.
    changed = [finished for finished in range(1, 16) if sent[finished][: len(sent[finished - 1]) - 1] != sent[finished - 1][:-1]]
    assert changed == [6, 9, 12, 15]


def test_a_forks_reply_is_one_line_and_one_that_is_not_an_answer_is_refused() -> None:
    assert sentence(said("  The file\nholds 400 zeros. ")) == "The file holds 400 zeros."
    with pytest.raises(ValueError):
        sentence(said(" \n"))
    with pytest.raises(ValueError):
        sentence(Message("m", "claude-opus-5-5", (ToolUse("t", "Read", {}),), "tool_use", {}))


def test_a_question_names_the_call_by_tool_id_input_and_the_ends_of_its_result_and_the_key_is_what_came_back() -> None:
    result = Result("call7", "Read", {"file_path": "/7"}, "head " + "7" * LONG + " tail", 7)
    assert "Read call call7 with input {\"file_path\": \"/7\"}" in question(result)
    # Two reads of one thing a turn apart are told apart by what each came back with.
    assert question(result) != question(Result("call7", "Read", {"file_path": "/7"}, "head " + "7" * LONG + " later", 7))
    assert key(result) == key(Result("call9", "Read", {"file_path": "/7"}, result.text, 2))
    assert key(result) != key(Result("call7", "Read", {"file_path": "/7"}, "8" * LONG, 7))


class Brain:
    """A brain whose forks send their request on the wire, from its history as it stands, before they answer."""

    session = BRAIN

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.failing: set[str] = set()
        self.history: dict[str, object] = history(0)
        self.wire: Callable[[Observed], None] = lambda _observed: None
        # What the fork's model replies over the wire, given the question.
        self.reply: Callable[[str], Message] = lambda question: said(f"the read of {question.split(' call ', 1)[1].split(' ', 1)[0]} said its digit")

    async def fork(self, question: str) -> str:
        self.asked.append(question)
        exchange = f"fork{len(self.asked)}"
        messages = cast(list[object], self.history["messages"])
        body = {**self.history, "messages": [*messages[:-1], prompt(f"{SIDE_QUESTION_OPENING} ...</system-reminder>\n\n{question}", marked=True)]}
        if any(call in question for call in self.failing):
            # Refused by Claude Code before any request of its own.
            raise ForkFailed("no snapshot")
        self.wire(Sent(exchange, BRAIN, classify("/v1/messages", body), body))
        message = self.reply(question)
        self.wire(Exchanged(exchange, BRAIN, Fork(), "POST", "/v1/messages", 1, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 1, Streamed(message))))
        return "what Claude Code hands back, which is not kept"


def said(text: str, stop_reason: str = "end_turn") -> Message:
    return Message("m", "claude-opus-5-5", (Text(text),), stop_reason, {})


class Store:
    def __init__(self) -> None:
        self.said: dict[Digest, str] = {}

    def known(self, digest: Digest) -> str | None:
        return self.said.get(digest)

    def keep(self, said: Mapping[Digest, str]) -> None:
        self.said.update(said)


class Stage:
    """A stage that sends every request of the brain's with a tail."""

    def route(self, sent: Sent) -> Route:
        return Send((Tail("[hands] tail"),)) if isinstance(sent.kind, MainTurn) else Send()

    def hear(self, observed: object) -> None:
        pass


def sent(body: object, kind: Kind | None = None, session: SessionId = BRAIN) -> Sent:
    return Sent("x", session, kind or classify("/v1/messages", body), body)


def ended(kind: Kind = MainTurn()) -> Exchanged:
    message = Message("m", "claude-opus-5-5", (Text("done"),), "end_turn", {})
    return Exchanged("x", BRAIN, kind, "POST", "/v1/messages", 1, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 1, Streamed(message)))


class Rig:
    def __init__(self, every: int = 3) -> None:
        self.brain = Brain()
        self.store = Store()
        self.recorded: list[Entry] = []
        self.keeper = Keeper(self.brain, self.store, every, self.recorded.append)
        self.kept = Kept(Stage(), self.keeper)
        self.brain.wire = self.kept.hear

    async def turn(self, finished: int) -> Route:
        """The brain's main turn after `finished` turns: routed, heard, and ended, and its results asked about."""
        self.brain.history = history(finished)
        request = sent(self.brain.history)
        self.kept.hear(request)
        route = self.kept.route(request)
        self.kept.hear(ended())
        # The keeper's worker takes whatever the turn's end queued.
        for _ in range(20):
            await asyncio.sleep(0)
        return route


@pytest.fixture
async def rig() -> Rig:
    return Rig()


async def test_each_result_is_asked_of_a_fork_once_after_its_turn_ends_and_goes_as_its_line_once_its_batch_comes(rig: Rig) -> None:
    worker = asyncio.create_task(rig.keeper.keep_asking())
    try:
        routes = [await rig.turn(finished) for finished in range(0, 8)]
    finally:
        worker.cancel()
    # Each result asked about once, in the order its turn ended.
    assert [question.split(" call ", 1)[1].split(" ", 1)[0] for question in rig.brain.asked] == [f"call{n}" for n in range(7)]
    stubs = {finished: [change for change in route.changes if isinstance(change, Stub)] for finished, route in enumerate(routes) if isinstance(route, Send)}
    assert all(stubs[finished] == [] for finished in range(0, 6))
    assert stubs[6] == stubs[7] == [Stub(f"call{n}", f"Read: the read of call{n} said its digit") for n in range(3)]
    # The stubs go before the stage's own tail.
    assert routes[7] == Send((*stubs[7], Tail("[hands] tail")))
    assert [entry for entry in rig.recorded if isinstance(entry, ResultsStubbed)] == [ResultsStubbed(("call0", "call1", "call2"), ())]


async def test_a_result_with_no_sentence_when_its_batch_comes_goes_whole_for_good_and_says_so(rig: Rig) -> None:
    rig.brain.failing.add("call1")
    errors: list[str] = []
    from loguru import logger

    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    worker = asyncio.create_task(rig.keeper.keep_asking())
    try:
        for finished in range(0, 7):
            await rig.turn(finished)
        # Said later, by any means: the batch was decided without it and stays as it went.
        rig.store.said[key(aged(history(6), 3)[1])] = "late"
        route = await rig.turn(7)
    finally:
        worker.cancel()
        logger.remove(sink)
    assert isinstance(route, Send) and [change.call for change in route.changes if isinstance(change, Stub)] == ["call0", "call2"]
    assert [entry for entry in rig.recorded if isinstance(entry, ResultsStubbed)] == [ResultsStubbed(("call0", "call2"), ("call1",))]
    assert [error for error in errors if "call1" in error] == ["no sentence for Read call call1: no snapshot"]


async def test_forks_and_compaction_share_the_stubs_and_only_main_turns_move_the_batch(rig: Rig) -> None:
    rig.store.said.update({key(result): f"said {result.call}" for result in aged(history(6), 3)})
    main = rig.kept.route(sent(history(6)))
    assert isinstance(main, Send)
    stubs = [change for change in main.changes if isinstance(change, Stub)]
    assert len(stubs) == 3
    question = SIDE_QUESTION_OPENING + " ...</system-reminder>\n\nwhat did it say?"
    # Past the next boundary by its own count, the side question still goes with the main turn's stubs and no more.
    side = history(9)
    side["messages"] = [*side["messages"], prompt(question)]  # pyright: ignore
    assert rig.kept.route(sent(side)) == Send(tuple(stubs))
    compacting = history(9)
    kept = COMPACTION_INSTRUCTIONS + "keep the draft" + COMPACTION_REMINDER + " Respond with plain text only."
    compacting["messages"] = [*compacting["messages"][:-1], prompt(COMPACTION_OPENING + " summarise the code" + kept)]  # pyright: ignore
    compaction = rig.kept.route(sent(compacting))
    assert compaction == Send((*stubs, Steer(VOICE_COMPACTION)))
    assert isinstance(sent(compacting).kind, Compaction) and isinstance(sent(side).kind, Fork)
    # The steered prompt is what goes, with what Claude Code wrote after its own, and the request is still a compaction.
    assert isinstance(compaction, Send)
    steered = edited(compacting, compaction.changes)
    assert classify("/v1/messages", steered) == Compaction()
    assert cast(list[dict[str, list[dict[str, str]]]], steered["messages"])[-1]["content"][0]["text"] == VOICE_COMPACTION + kept


async def test_another_sessions_requests_and_count_tokens_go_unchanged(rig: Rig) -> None:
    rig.store.said.update({key(result): "said" for result in aged(history(6), 3)})
    assert rig.keeper.changes(sent(history(6), session=SessionId("work"))) == ()
    assert rig.keeper.changes(sent(history(6), kind=CountTokens())) == ()


async def test_a_held_request_goes_held_whatever_the_keeper_would_change() -> None:
    class Holding(Stage):
        def route(self, sent: Sent) -> Route:
            return Hold("(stayed silent)")

    keeper = Keeper(Brain(), Store(), 3, lambda _entry: None)
    assert Kept(Holding(), keeper).route(sent(history(6))) == Hold("(stayed silent)")


async def test_a_sentence_is_kept_only_when_the_forks_own_request_held_the_result_whole(rig: Rig) -> None:
    errors: list[str] = []
    from loguru import logger

    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    await rig.turn(0)
    # A compaction lands before the queue drains: the brain's history is its summary, and the forks see no results.
    request = sent(history(1))
    rig.kept.hear(request)
    rig.kept.route(request)
    rig.brain.history = {**history(0), "messages": [prompt("the summary", marked=True)]}
    rig.kept.hear(ended())
    worker = asyncio.create_task(rig.keeper.keep_asking())
    try:
        for _ in range(20):
            await asyncio.sleep(0)
    finally:
        worker.cancel()
        logger.remove(sink)
    assert rig.brain.asked and rig.store.said == {}
    assert errors == ["no sentence for Read call call0: the fork's request did not hold the result whole"]


async def test_what_claude_code_says_for_a_fork_whose_request_failed_is_never_kept(rig: Rig) -> None:
    # Claude Code hands back "(API error: ...)" as a fork's answer; on the wire the model's reply never ended in words.
    rig.brain.reply = lambda _question: said("", stop_reason="max_tokens")
    worker = asyncio.create_task(rig.keeper.keep_asking())
    try:
        await rig.turn(0)
        await rig.turn(1)
    finally:
        worker.cancel()
    assert rig.brain.asked and rig.store.said == {}
