"""Each call of a tool is one wide event, through whichever adapter called it: returned, raised, or refused."""

from typing import Literal, TypedDict

import pytest

from hands.sessions.audit import Entry
from hands.sessions.wide import WideEvent
from hands.voice.tools import Called, Result, Tool, audited, tool


async def echo(text: str) -> Result:
    """Say it back.

    Args:
        text: What to say.
    """
    return {"said": text}


async def refusing(session: str) -> Result:
    """Refuse.

    Args:
        session: Whose.
    """
    return {"error": f"no running session has the id {session!r}"}


class Line(TypedDict):
    text: str
    loud: bool


async def typed(text: str, loud: bool, times: int, tone: Literal["flat", "bright"], lines: list[Line]) -> Result:
    """Say it, as asked.

    Args:
        text: What to say.
        loud: Whether to say it loudly.
        times: How many times.
        tone: The tone to say it in.
        lines: More to say.
    """
    raise AssertionError("a call whose arguments do not fit never reaches the body")


async def broken(session: str) -> Result:
    """Fail.

    Args:
        session: Whose.
    """
    raise RuntimeError("the transcript went away")


def run(recorded: list[Entry]) -> WideEvent:
    [event] = recorded
    assert isinstance(event, WideEvent) and event.event == "tool.run"
    return event


async def test_an_audited_tool_keeps_its_schema() -> None:
    plain = tool(echo)
    wrapped = audited(plain, lambda _: None)
    assert (wrapped.name, wrapped.description, wrapped.input_schema, wrapped.completes) == (plain.name, plain.description, plain.input_schema, plain.completes)


async def test_a_call_that_returns_is_one_event_with_what_it_was_given_and_handed_back() -> None:
    recorded: list[Entry] = []
    assert await audited(tool(echo), recorded.append).body(text="hi") == {"said": "hi"}
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("ok", None, {"tool": "echo", "called": Called({"text": "hi"}, {"said": "hi"})})


async def test_a_call_whose_result_refuses_it_is_one_failed_event_saying_why() -> None:
    recorded: list[Entry] = []
    result = await audited(tool(refusing), recorded.append).body(session="x")
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("failed", "no running session has the id 'x'", {"tool": "refusing", "called": Called({"session": "x"}, result)})


async def test_a_call_with_arguments_the_body_does_not_take_is_refused_to_the_model_and_one_failed_event() -> None:
    recorded: list[Entry] = []
    body = tool(echo)
    result = await audited(body, recorded.append).body(words="hi")
    assert result == {"error": "echo was called with the wrong arguments: missing a required argument: 'text'"}
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("failed", result["error"], {"tool": "echo", "called": Called({"words": "hi"}, result)})


FITTING: dict[str, object] = {"text": "hi", "loud": False, "times": 2, "tone": "flat", "lines": [{"text": "more", "loud": True}]}


@pytest.mark.parametrize(
    ("given", "refusal"),
    [
        # The truthiness of a string is not what the model said: 'false' would have been taken as true.
        ({"loud": "false"}, "loud should be true or false, got 'false'"),
        ({"times": "3"}, "times should be an integer, got '3'"),
        ({"times": True}, "times should be an integer, got True"),
        ({"text": 3}, "text should be a string, got 3"),
        ({"text": None}, "text should be a string, got None"),
        ({"tone": "dull"}, "'dull' is no tone; it is one of flat, bright"),
        ({"lines": "more"}, "lines should be a list, got 'more'"),
        ({"lines": ["more"]}, "lines[0] should be an object, got 'more'"),
        ({"lines": [{"text": "more"}]}, "lines[0] has no loud"),
        ({"lines": [{"text": "more", "loud": "yes"}]}, "lines[0].loud should be true or false, got 'yes'"),
        ({"loud": "false", "times": "3"}, "loud should be true or false, got 'false'; times should be an integer, got '3'"),
    ],
)
async def test_an_argument_that_is_not_what_the_schema_says_is_refused_before_the_body_runs_and_one_failed_event(given: dict[str, object], refusal: str) -> None:
    recorded: list[Entry] = []
    arguments = {**FITTING, **given}
    result = await audited(tool(typed), recorded.append).body(**arguments)
    assert result == {"error": refusal}
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("failed", refusal, {"tool": "typed", "called": Called(arguments, result)})


async def test_arguments_that_are_what_the_schema_says_reach_the_body() -> None:
    with pytest.raises(AssertionError, match="never reaches the body"):
        await tool(typed).body(**FITTING)


async def test_a_call_that_raises_is_one_failed_event_with_what_raised_and_where_and_raises_on() -> None:
    recorded: list[Entry] = []
    wrapped: Tool = audited(tool(broken), recorded.append)
    with pytest.raises(RuntimeError):
        await wrapped.body(session="s1")
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("failed", "RuntimeError: the transcript went away", {"tool": "broken", "called": Called({"session": "s1"}, None)})
    # Where it was raised, a frame of the test's own, is the last frame.
    assert event.trace[-1].endswith(" in broken")
