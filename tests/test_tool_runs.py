"""Each call of a tool is one wide event, through whichever adapter called it: returned, raised, or refused."""

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


async def test_a_call_that_raises_is_one_failed_event_with_what_raised_and_where_and_raises_on() -> None:
    recorded: list[Entry] = []
    wrapped: Tool = audited(tool(broken), recorded.append)
    with pytest.raises(RuntimeError):
        await wrapped.body(session="s1")
    event = run(recorded)
    assert (event.outcome, event.error, event.facts) == ("failed", "RuntimeError: the transcript went away", {"tool": "broken", "called": Called({"session": "s1"}, None)})
    # Where it was raised, a frame of the test's own, is the last frame.
    assert event.trace[-1].endswith(" in broken")
