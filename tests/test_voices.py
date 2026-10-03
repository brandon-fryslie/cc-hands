"""Choosing the voice: the voices heard in turn, the one kept, and what the speaker says each line in."""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pipecat.frames.frames import Frame, InterruptionFrame, TTSSpeakFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from conftest import running
from hands.sessions.audit import Called, Entry
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice import voices
from hands.voice.tools import audited, voice_tools
from hands.voice.voices import DEFAULT, VOICES, Voice, Voices, chosen, parse_voice, sample


class Speaker(TTSService):
    """Pipecat's own TTS service with the synthesis taken out: each line is said, as a record of the voice it was said
    in, with the service's settings moved by the same frames that move pocket-tts's."""

    def __init__(self, voice: Voice) -> None:
        super().__init__(settings=PocketTTSSettings(model=None, voice=voice, language=Language.EN))  # pyright: ignore[reportUnknownMemberType]
        self.said: list[tuple[str, str]] = []
        self.spoke = asyncio.Event()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.said.append((str(self._settings.voice), text))
        self.spoke.set()
        return
        yield

    async def until(self, count: int) -> list[tuple[str, str]]:
        while len(self.said) < count:
            self.spoke.clear()
            await asyncio.wait_for(self.spoke.wait(), 2.0)
        return self.said


def ready(voice: Voice) -> None:
    """Every voice already in the cache."""


def downloading(voice: Voice) -> None:
    """Every voice fetched as a download is: taking long enough for another call to run meanwhile."""
    time.sleep(0.05)


def offline(voice: Voice) -> None:
    raise OSError(f"cannot reach the hub for {voice}")


def test_a_voice_is_named_as_a_person_says_it_and_one_hands_does_not_have_is_refused() -> None:
    assert parse_voice("Bill Boerst") == "bill_boerst"
    assert parse_voice("charles") == DEFAULT
    with pytest.raises(Rejected, match="no voice called 'Zed'"):
        parse_voice("Zed")


def test_the_voice_is_the_default_until_one_is_kept() -> None:
    home = Home(Path("/nonexistent"))
    assert chosen(home) == DEFAULT


def test_a_chosen_voice_is_read_back_by_a_daemon_started_after_it(tmp_path: Path) -> None:
    voices.keep(Home(tmp_path), parse_voice("Mary"))
    # A new Home over the same directory is a restart: nothing is carried but the file.
    assert chosen(Home(tmp_path)) == "mary"


def test_a_hand_edited_choice_hands_has_no_voice_for_is_refused_by_name(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.voice.write_text("zed\n")
    with pytest.raises(Rejected, match=f"{home.voice} says 'zed'"):
        chosen(home)


def test_a_choice_that_is_not_text_is_refused_by_name(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.voice.write_bytes(b"\xff\xfe")
    with pytest.raises(Rejected, match="names no voice"):
        chosen(home)


async def test_a_voice_that_cannot_be_fetched_is_neither_heard_nor_kept(tmp_path: Path) -> None:
    home = Home(tmp_path)
    lines, speaker = IdentityFilter(), Speaker(DEFAULT)
    tools = {tool.name: tool for tool in voice_tools(Voices(home, lines, offline))}
    async with running([lines, speaker]) as run:
        assert "cannot reach the hub for mary" in str((await tools["hear_voices"].body(names=["Mary"]))["error"])
        assert "cannot reach the hub for mary" in str((await tools["use_voice"].body(name="Mary"))["error"])
        await run.worker.queue_frame(TTSSpeakFrame("Still me."))
        assert await speaker.until(1) == [(DEFAULT, "Still me.")]
    assert not home.voice.exists()


async def test_the_voices_heard_each_say_their_line_in_their_own_voice_and_the_next_line_is_in_the_kept_one(tmp_path: Path) -> None:
    home = Home(tmp_path)
    lines, speaker = IdentityFilter(), Speaker(DEFAULT)
    chosen_voices = Voices(home, lines, ready)
    async with running([lines, speaker]) as run:
        assert await chosen_voices.hear((Voice("mary"), Voice("bill_boerst"))) == DEFAULT
        await run.worker.queue_frame(TTSSpeakFrame("Back to you."))
        assert await speaker.until(3) == [("mary", sample(Voice("mary"))), ("bill_boerst", sample(Voice("bill_boerst"))), (DEFAULT, "Back to you.")]

        await chosen_voices.use(Voice("mary"))
        await run.worker.queue_frame(TTSSpeakFrame("Like this."))
        assert (await speaker.until(4))[-1] == ("mary", "Like this.")
    assert chosen(Home(tmp_path)) == "mary"


async def test_a_hearing_and_a_choice_called_together_each_run_whole(tmp_path: Path) -> None:
    home = Home(tmp_path)
    lines, speaker = IdentityFilter(), Speaker(DEFAULT)
    chosen_voices = Voices(home, lines, downloading)
    async with running([lines, speaker]) as run:
        # As Pipecat runs two calls of one reply: side by side.
        await asyncio.gather(chosen_voices.hear((Voice("alba"), Voice("bill_boerst"))), chosen_voices.use(Voice("mary")))
        await run.worker.queue_frame(TTSSpeakFrame("After both."))
        said = await speaker.until(3)
    assert said == [("alba", sample(Voice("alba"))), ("bill_boerst", sample(Voice("bill_boerst"))), ("mary", "After both.")]
    assert chosen(home) == "mary"


async def test_a_hearing_cut_short_still_ends_in_the_voice_hands_speaks_in(tmp_path: Path) -> None:
    home = Home(tmp_path)
    lines, speaker = IdentityFilter(), Speaker(DEFAULT)
    async with running([lines, speaker]) as run:
        await Voices(home, lines, ready).hear(VOICES[:3])
        await run.worker.queue_frame(InterruptionFrame())
        await run.worker.queue_frame(TTSSpeakFrame("After the barge-in."))
        said = await speaker.until(1)
        while said[-1][1] != "After the barge-in.":
            said = await speaker.until(len(said) + 1)
    assert said[-1] == (DEFAULT, "After the barge-in.")


async def test_each_voice_tool_answers_with_the_voice_spoken_in_after_it_and_refuses_a_voice_hands_lacks(tmp_path: Path) -> None:
    recorded: list[Entry] = []
    lines, speaker = IdentityFilter(), Speaker(DEFAULT)
    tools = {tool.name: audited(tool, recorded.append) for tool in voice_tools(Voices(Home(tmp_path), lines, ready))}
    async with running([lines, speaker]):
        assert await tools["voices_on_offer"].body() == {"speaking_in": DEFAULT, "voices": VOICES}
        assert await tools["hear_voices"].body(names=["Eve"]) == {"heard": ("eve",), "speaking_in": DEFAULT}
        assert await tools["use_voice"].body(name="Eve") == {"speaking_in": "eve", "readback": "This is Eve, and I'll speak in this voice from now on."}
        assert (await tools["voices_on_offer"].body())["speaking_in"] == "eve"
        refused = await tools["use_voice"].body(name="Zed")
        assert "no voice called 'Zed'" in str(refused["error"])
    assert [type(entry) for entry in recorded] == [Called] * 5
    assert chosen(Home(tmp_path)) == "eve"
