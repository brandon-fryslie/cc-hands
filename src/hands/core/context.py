"""The brain's context kept small by rule: which tool results are old enough to go as one line, and what the line says.

A long result goes as one line once it is more than `every` turns old, and results go in batches of `every` turns, so
the history the API's cache holds changes once a batch and not every turn. What the line says is asked of a fork of the
brain while the result is still whole in its prefix (`hands.brain.context`), and kept in the summary store by the
result's key, so it is asked once whatever becomes of the brain.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass

from hands.core.sentences import Digest, digest
from hands.core.wire import tool_answers, tool_calls, turns

# The summariser's version in every result's key: a change to how a result is asked about is a new key for each.
VERSION = "tool-result-1"
# A result shorter than this is about as short as its sentence would be: it goes whole, and no fork is asked about it.
LONG = 400
# How much of a call's input a question quotes to name the call: enough to tell two reads apart.
QUOTED = 300
# What Claude Code answers a side question with when the model called a tool instead, in its utils/sideQuestion.ts.
TRIED_A_TOOL = "(The model tried to call "


@dataclass(frozen=True)
class Result:
    """A long tool result in a request's history: the call it answers, what was called with what, its text, and its turn."""

    call: str
    tool: str
    input: Mapping[str, object]
    text: str
    turn: int


def results(body: object) -> tuple[Result, ...]:
    """Every long tool result a request carries, oldest first; raises ValueError for one that answers no call in it,
    which the API would refuse."""
    calls = tool_calls(body)
    answers = [answer for answer in tool_answers(body) if len(answer.text) >= LONG]
    if unanswered := [answer.call for answer in answers if answer.call not in calls]:
        raise ValueError(f"tool results answer calls the history does not hold: {unanswered}")
    return tuple(Result(answer.call, calls[answer.call].name, calls[answer.call].input, answer.text, answer.turn) for answer in answers)


def boundary(held: int, every: int) -> int:
    """The first turn whose results go whole in a request holding `held` turns: every result before it is more than
    `every` turns old, and it moves only once every `every` turns."""
    return max(0, ((held - 1) // every - 1) * every)


def aged(body: object, every: int) -> tuple[Result, ...]:
    """The long results a request carries from before its boundary: the ones that go as a line."""
    edge = boundary(turns(body), every)
    return tuple(result for result in results(body) if result.turn < edge)


def key(result: Result) -> Digest:
    """What a result's sentence is kept under: what was called, with what, and what came back."""
    return digest(VERSION, json.dumps([result.tool, result.input, result.text], sort_keys=True), ())


def question(result: Result) -> str:
    """What a fork of the brain is asked about a result, which its prefix still holds whole."""
    given = json.dumps(result.input, sort_keys=True)
    quoted = given if len(given) <= QUOTED else given[:QUOTED] + "…"
    return (
        f"In one sentence of at most 30 words, say what the result of your {result.tool} call {result.call} with input "
        f"{quoted} told you, so that the sentence can stand in for the result from now on. Answer with the sentence alone."
    )


def sentence(reply: str) -> str:
    """A fork's reply as the one line that stands in for a result; raises ValueError for a reply that is not one."""
    said = " ".join(reply.split())
    if not said:
        raise ValueError("the fork answered nothing")
    if said.startswith(TRIED_A_TOOL):
        raise ValueError(f"the fork called a tool instead of answering: {said}")
    return said


def line(result: Result, said: str) -> str:
    """The line a result goes as once it is old: what was called, and what it told the brain."""
    return f"{result.tool}: {said}"
