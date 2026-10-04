import asyncio

import pytest

from hands.voice.tools import trigger_tools
from hands.voice.trigger import Trigger, Triggers


async def test_the_trigger_in_use_is_said_and_starts_as_the_held_key() -> None:
    in_use, _ = trigger_tools(Triggers())
    assert await in_use.body() == {"trigger": "held key", "readback": "The held key: hold Right Shift to talk, and let go to send."}


async def test_a_trigger_not_built_is_refused_and_the_one_in_use_stays() -> None:
    triggers = Triggers()
    _, switch = trigger_tools(triggers)
    assert await switch.body(trigger="wake word") == {"error": "'wake word' is no trigger; it is one of held key"}
    assert triggers.in_use == "held key"


async def test_set_trigger_to_the_one_in_use_says_it_is_already_on() -> None:
    triggers = Triggers()
    _, switch = trigger_tools(triggers)
    assert await switch.body(trigger="held key") == {"trigger": "held key", "was": "held key", "readback": "Already on. The held key: hold Right Shift to talk, and let go to send."}
    assert triggers.in_use == "held key"


async def test_the_desk_is_driven_by_the_edge_of_the_trigger_in_use_and_choosing_it_again_does_not_restart_it() -> None:
    triggers = Triggers()
    started: list[str] = []
    stopped: list[str] = []

    async def edge(trigger: Trigger) -> None:
        started.append(trigger)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(trigger)

    driving = asyncio.create_task(triggers.drive(edge))
    while not started:
        await asyncio.sleep(0)
    triggers.choose("held key")  # a hold under way must not be dropped by a switch to the trigger already on
    for _ in range(10):
        await asyncio.sleep(0)
    assert (started, stopped) == (["held key"], [])
    driving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await driving
    assert stopped == ["held key"]


async def test_an_edge_that_fails_ends_the_drive_with_its_failure() -> None:
    async def edge(_trigger: Trigger) -> None:
        raise OSError("the event tap was refused")

    with pytest.raises(ExceptionGroup) as failed:
        await Triggers().drive(edge)
    assert failed.group_contains(OSError, match="the event tap was refused")
