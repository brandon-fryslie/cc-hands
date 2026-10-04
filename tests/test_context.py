"""The brain's context kept from the proxy: old results as a line, in batches, from sentences asked for as side questions that show the results."""

import asyncio
import re
from collections.abc import Mapping
from typing import cast

import pytest

from hands.brain.context import VOICE_COMPACTION, Keeper, Kept
from hands.brain.asides import AsideFailed, Unanswered
from hands.core.context import LONG, QUOTED, SHOWN, Result, aged, boundary, key, line, question
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
    Observed,
    Reached,
    Route,
    Send,
    Sent,
    Steer,
    Streamed,
    Stub,
    Tail,
    Text,
    classify,
    edited,
)
from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent, root

BRAIN = SessionId("brain")
# The span the stages here route with: which trace a request is in is the stage's, not the keeper's.
SPAN = root()
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


def test_a_question_shows_the_call_and_its_result_a_long_one_by_its_ends_and_the_key_is_what_came_back() -> None:
    result = Result("call7", "Read", {"file_path": "/7"}, "head " + "7" * LONG + " tail", 7)
    assert 'Tool: Read\nInput: {"file_path": "/7"}\n\n<recorded_result>\n' + result.text + "\n</recorded_result>" in question(result)
    # The question stands alone: whoever answers it has no conversation, so it names no call of theirs.
    assert "call7" not in question(result)
    long = Result("call8", "Bash", {"command": "x" * 400}, "head " + "8" * 2 * SHOWN + " tail", 8)
    assert f"head {'8' * (SHOWN // 2 - 5)}\n[{len(long.text) - SHOWN} characters left out]\n{'8' * (SHOWN // 2 - 5)} tail\n</recorded_result>" in question(long)
    assert 'Input: {"command": "' + "x" * (QUOTED - len('{"command": "')) + "…\n" in question(long)
    assert key(result) == key(Result("call9", "Read", {"file_path": "/7"}, result.text, 2))
    assert key(result) != key(Result("call7", "Read", {"file_path": "/7"}, "8" * LONG, 7))


class Asked:
    """What answers the keeper's side questions: a sentence for the read each one shows."""

    def __init__(self) -> None:
        self.files: list[str] = []
        self.failing: set[str] = set()

    async def ask(self, question: str) -> str:
        [file] = re.findall(r'Input: \{"file_path": "(/\d+)"\}', question)
        self.files.append(file)
        if file in self.failing:
            raise AsideFailed(Unanswered.TIMED_OUT, "no answer in 120s")
        # Broken over lines, as a model may break it.
        return f"  file {file}\nholds its digit. "


class Store:
    def __init__(self) -> None:
        self.said: dict[Digest, str] = {}

    def known(self, digest: Digest) -> str | None:
        return self.said.get(digest)

    def keep(self, said: Mapping[Digest, str]) -> None:
        self.said.update(said)


class NoAsides:
    """No side question asked: every request goes as the stage routed it."""

    def adopted(self, sent: Sent, routed: Route) -> Route:
        return routed

    def hear(self, observed: Observed) -> None:
        pass


class Stage:
    """A stage that sends every request of the brain's with a tail."""

    def route(self, sent: Sent) -> Route:
        return Send((Tail("[hands] tail"),), span=SPAN) if isinstance(sent.kind, MainTurn) else Send(span=SPAN)

    def hear(self, observed: object) -> None:
        pass


def sent(body: object, kind: Kind | None = None, session: SessionId = BRAIN) -> Sent:
    return Sent("x", session, kind or classify("/v1/messages", body), body)


def ended(kind: Kind = MainTurn(None)) -> Exchanged:
    message = Message("m", "claude-opus-5-5", (Text("done"),), "end_turn", {})
    return Exchanged("x", BRAIN, kind, "POST", "/v1/messages", 1, (), 0.0, 0.0, Reached(200, 0.0, 0.0, 1, Streamed(message)), False, root())


class Rig:
    def __init__(self, every: int = 3) -> None:
        self.asked = Asked()
        self.store = Store()
        self.recorded: list[Entry] = []
        self.keeper = Keeper(BRAIN, self.asked.ask, self.store, every, self.recorded.append)
        self.kept = Kept(Stage(), self.keeper, NoAsides())

    async def turn(self, finished: int) -> Route:
        """The brain's main turn after `finished` turns: routed, heard, and ended, and its results asked about."""
        request = sent(history(finished))
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


def stubbings(recorded: list[Entry]) -> list[tuple[Mapping[str, object], Mapping[str, int]]]:
    """Each batch reached: its facts and its counts."""
    return [(entry.facts, entry.counts) for entry in recorded if isinstance(entry, WideEvent) and entry.event == "context.stubbing"]


async def test_each_result_is_asked_about_once_after_its_turn_ends_and_goes_as_its_line_once_its_batch_comes(rig: Rig) -> None:
    worker = asyncio.create_task(rig.keeper.keep_asking())
    try:
        routes = [await rig.turn(finished) for finished in range(0, 8)]
    finally:
        worker.cancel()
    # Each result asked about once, in the order its turn ended.
    assert rig.asked.files == [f"/{n}" for n in range(7)]
    stubs = {finished: [change for change in route.changes if isinstance(change, Stub)] for finished, route in enumerate(routes) if isinstance(route, Send)}
    assert all(stubs[finished] == [] for finished in range(0, 6))
    assert stubs[6] == stubs[7] == [Stub(f"call{n}", f"Read: file /{n} holds its digit.") for n in range(3)]
    # The stubs go before the stage's own tail.
    assert routes[7] == Send((*stubs[7], Tail("[hands] tail")), span=SPAN)
    assert stubbings(rig.recorded) == [({"stubbed": ("call0", "call1", "call2"), "whole": ()}, {"stubbed": 3, "whole": 0})]


async def test_a_result_with_no_sentence_when_its_batch_comes_goes_whole_for_good_and_says_so(rig: Rig) -> None:
    rig.asked.failing.add("/1")
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
    assert stubbings(rig.recorded) == [({"stubbed": ("call0", "call2"), "whole": ("call1",)}, {"stubbed": 2, "whole": 1})]
    assert [error for error in errors if "call1" in error] == ["no sentence for Read call call1: no answer in 120s"]


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
    assert rig.kept.route(sent(side)) == Send(tuple(stubs), span=SPAN)
    compacting = history(9)
    kept = COMPACTION_INSTRUCTIONS + "keep the draft" + COMPACTION_REMINDER + " Respond with plain text only."
    compacting["messages"] = [*compacting["messages"][:-1], prompt(COMPACTION_OPENING + " summarise the code" + kept)]  # pyright: ignore
    compaction = rig.kept.route(sent(compacting))
    assert compaction == Send((*stubs, Steer(VOICE_COMPACTION)), span=SPAN)
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
            return Hold("(stayed silent)", SPAN)

    keeper = Keeper(BRAIN, Asked().ask, Store(), 3, lambda _entry: None)
    assert Kept(Holding(), keeper, NoAsides()).route(sent(history(6))) == Hold("(stayed silent)", SPAN)


async def test_the_stages_send_goes_on_whole_with_the_keepers_changes_first() -> None:
    class Final(Stage):
        def route(self, sent: Sent) -> Route:
            return Send((Tail("standing"),), refusal="final", span=SPAN)

    store = Store()
    store.said.update({key(result): "said" for result in aged(history(6), 3)})
    keeper = Keeper(BRAIN, Asked().ask, store, 3, lambda _entry: None)
    routed = Kept(Final(), keeper, NoAsides()).route(sent(history(6)))
    assert isinstance(routed, Send) and routed.refusal == "final"
    assert routed.changes[-1] == Tail("standing") and len(routed.changes) > 1
