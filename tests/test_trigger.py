import asyncio
from pathlib import Path

import pytest

from hands.voice.tools import trigger_tools
from hands.voice.trigger import Trigger, Triggers
from hands.voice.wake import EMBEDDING, MELSPECTROGRAM
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained


async def test_the_trigger_in_use_is_said_and_starts_as_the_held_key(tmp_path: Path) -> None:
    in_use, _ = trigger_tools(Triggers(Pretrained()), tmp_path)
    assert await in_use.body() == {"trigger": "held key", "readback": "The held key: hold Right Shift to talk, and let go to send."}


async def test_a_trigger_not_built_is_refused_and_the_one_in_use_stays(tmp_path: Path) -> None:
    triggers = Triggers(Pretrained())
    _, switch = trigger_tools(triggers, tmp_path)
    assert await switch.body(trigger="clap") == {"error": "'clap' is no trigger; it is one of held key, engaged conversation, wake word"}
    assert triggers.in_use == "held key"


async def test_set_trigger_to_the_one_in_use_says_it_is_already_on(tmp_path: Path) -> None:
    triggers = Triggers(Pretrained())
    _, switch = trigger_tools(triggers, tmp_path)
    assert await switch.body(trigger="held key") == {"trigger": "held key", "was": "held key", "fetched": [], "readback": "Already on. The held key: hold Right Shift to talk, and let go to send."}
    assert triggers.in_use == "held key"


async def test_the_desk_is_driven_by_the_edge_of_the_trigger_in_use_and_choosing_it_again_does_not_restart_it() -> None:
    triggers = Triggers(Pretrained())
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
        await Triggers(Pretrained()).drive(edge)
    assert failed.group_contains(OSError, match="the event tap was refused")


async def test_engaged_conversation_is_switched_to_and_says_how_to_talk_under_it(tmp_path: Path) -> None:
    triggers = Triggers(Pretrained())
    in_use, switch = trigger_tools(triggers, tmp_path)
    assert (await switch.body(trigger="engaged conversation"))["was"] == "held key"
    assert await in_use.body() == {
        "trigger": "engaged conversation",
        "readback": "Engaged conversation: hold Right Shift once to engage, then just talk; hands answers when you finish, and listens again. Hold it once more to disengage.",
    }


async def test_the_wake_word_is_switched_to_and_says_how_to_talk_under_it(tmp_path: Path) -> None:
    triggers = Triggers(Pretrained())
    in_use, switch = trigger_tools(triggers, tmp_path)
    for name in (MELSPECTROGRAM, EMBEDDING, PRETRAINED["Hey Jarvis"]):
        (tmp_path / name).write_bytes(b"model")
    switched = await switch.body(trigger="wake word")
    assert (switched["was"], switched["fetched"]) == ("held key", [])
    assert await in_use.body() == {
        "trigger": "wake word",
        "readback": "The wake word: say Hey Jarvis, then what you want; hands answers when you finish. It cannot hear the wake word while it speaks.",
    }


async def test_a_switch_to_the_wake_word_whose_models_cannot_be_fetched_is_refused_and_the_trigger_stays(tmp_path: Path) -> None:
    triggers = Triggers(Pretrained())
    (tmp_path / "wake-word").write_text("no directory")
    _, switch = trigger_tools(triggers, tmp_path / "wake-word")
    refused = await switch.body(trigger="wake word")
    assert (refused["trigger"], refused["readback"]) == ("held key", "The wake word could not be set up, so the trigger stays as it was.")
    assert str(refused["error"]).startswith("wake word could not be readied: ")
    assert triggers.in_use == "held key"


async def test_the_wake_word_the_settings_name_is_the_one_said_and_readied(tmp_path: Path) -> None:
    trained = tmp_path / "hey_computer.onnx"
    trained.write_bytes(b"model")
    for name in (MELSPECTROGRAM, EMBEDDING, PRETRAINED["Alexa"]):
        (tmp_path / name).write_bytes(b"model")
    for word, phrase in ((Pretrained("Alexa"), "Alexa"), (Trained("Hey Computer", trained), "Hey Computer")):
        in_use, switch = trigger_tools(Triggers(word), tmp_path)
        # Every model the word is heard with is there already, so none is fetched: Hey Jarvis's is not, and is not needed.
        assert (await switch.body(trigger="wake word"))["fetched"] == []
        assert (await in_use.body())["readback"] == f"The wake word: say {phrase}, then what you want; hands answers when you finish. It cannot hear the wake word while it speaks."


async def test_a_switch_to_a_trained_wake_word_whose_model_is_not_there_is_refused_naming_where_it_was_looked_for(tmp_path: Path) -> None:
    for name in (MELSPECTROGRAM, EMBEDDING):
        (tmp_path / name).write_bytes(b"model")
    triggers = Triggers(Trained("Hey Computer", tmp_path / "gone.onnx"))
    _, switch = trigger_tools(triggers, tmp_path)
    refused = await switch.body(trigger="wake word")
    assert refused["error"] == f"wake word could not be readied: no wake word model at {tmp_path / 'gone.onnx'}"
    assert triggers.in_use == "held key"
