from hands.voice.tools import trigger_tools
from hands.voice.trigger import Triggers


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
