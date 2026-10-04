"""The transcript: each line read off the pipeline once, from the processor that makes it."""

from pipecat.frames.frames import InterruptionFrame, TranscriptionFrame
from pipecat.processors.filters.identity_filter import IdentityFilter

from conftest import running
from hands.voice.transcript import Cut, Heard, Line, Saying, Spoken, TranscriptObserver
from test_playback import Heard as Reached
from test_playback import ended, spoken


async def test_each_line_is_told_once_as_whisper_hears_it_and_as_the_speaker_plays_it() -> None:
    told: list[Line] = []
    # Whisper, a processor between it and the TTS service, the output transport, and what follows it: every frame
    # crosses each boundary, and each line is still told once.
    stt, between, output, reached = IdentityFilter(), IdentityFilter(), IdentityFilter(), Reached()
    async with running([stt, between, output, reached], [TranscriptObserver(stt, output, told.append)]) as run:
        for frame in (
            TranscriptionFrame("Is the parser fixed?", "me", "now"),
            spoken("It is."),
            ended("It is."),
            spoken("Its tests pass."),
        ):
            await run.worker.queue_frame(frame)
        await reached.until(2, type(spoken("")))
        await run.worker.queue_frame(InterruptionFrame())
        await reached.until(1, InterruptionFrame)
    assert told == [Heard("Is the parser fixed?"), Saying("It is."), Spoken(), Saying("Its tests pass."), Cut()]
