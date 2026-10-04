"""The transcript: each line read off the pipeline once, from the processor that makes it."""

from pipecat.frames.frames import AggregatedTextFrame, InterruptionFrame, OutputAudioRawFrame, TranscriptionFrame
from pipecat.processors.filters.identity_filter import IdentityFilter

from conftest import running
from hands.voice.transcript import Cut, Heard, Line, Saying, Spoken, TranscriptObserver
from test_playback import Heard as Reached
from test_playback import ended, spoken


def audio() -> OutputAudioRawFrame:
    return OutputAudioRawFrame(bytes(640), 16000, 1)


async def test_each_line_is_told_once_as_whisper_hears_it_and_as_the_speaker_plays_it() -> None:
    told: list[Line] = []
    # Whisper, a processor between it and the TTS service, the output transport, and what follows it: every frame
    # crosses each boundary, and each line is still told once.
    stt, between, output, reached = IdentityFilter(), IdentityFilter(), IdentityFilter(), Reached()
    observer = TranscriptObserver(stt, output, told.append)
    async with running([stt, between, output, reached], [observer]) as run:
        for frame in (
            TranscriptionFrame("Is the parser fixed?", "me", "now"),
            # The first sentence of a reply is let go before its audio is made: it starts with its sound.
            # Its 6 characters in two frames of 20 ms: 150 a second.
            spoken("It is."),
            audio(),
            audio(),
            ended("It is."),
            spoken("Its tests pass."),
            audio(),
            # Let go in the moment of the barge-in, its audio never written.
            spoken("All of them."),
        ):
            await run.worker.queue_frame(frame)
        # Every sentence let go, and the one ended, which is a kind of AggregatedTextFrame too.
        await reached.until(4, AggregatedTextFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await reached.until(1, InterruptionFrame)
        # After the barge-in, sound of the next reply, ahead of any sentence of it.
        await run.worker.queue_frame(audio())
        await reached.until(4, OutputAudioRawFrame)
    assert told == [Heard("Is the parser fixed?"), Saying("It is.", 15.0), Spoken(), Saying("Its tests pass.", 82.5), Cut()]
