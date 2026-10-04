"""The conversation as the audit log tells it: each reply written as the speaker finishes it, whoever's words it was."""

import asyncio
from collections.abc import AsyncGenerator, Sequence

from pipecat.frames.frames import Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from conftest import running
from hands.sessions.audit import Entry, Replied
from hands.voice.conversation import record_turns
from hands.voice.speech import as_a_turn
from hands.voice.spoken import FenceAggregator


class Speaker(TTSService):
    """The speaker with the synthesis taken out: started, stopped, and its text pushed by Pipecat's service as pocket-tts's are."""

    def __init__(self) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            push_start_frame=True, push_stop_frames=True, stop_frame_timeout_s=0.05, settings=PocketTTSSettings(model=None, voice="alba", language=Language.EN)
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield TTSAudioRawFrame(b"\x00\x00", self.sample_rate, 1, context_id=context_id)


async def replies(frames: Sequence[Frame], count: int) -> list[Entry]:
    """What the audit log is told of the frames, sent as the model's stage sends them, once `count` replies are written."""
    recorded: list[Entry] = []
    written = asyncio.Event()

    def record(entry: Entry) -> None:
        recorded.append(entry)
        written.set()

    pair = LLMContextAggregatorPair(LLMContext())
    record_turns(pair.user(), pair.assistant(), record)
    async with running([LLMTextProcessor(text_aggregator=FenceAggregator()), Speaker(), pair.assistant()]) as pipeline:
        await pipeline.worker.queue_frames(frames)
        while len(recorded) < count:
            written.clear()
            await asyncio.wait_for(written.wait(), 2.0)
    return recorded


async def test_a_readback_hands_says_after_the_models_reply_is_written_as_heard_whole_once_it_is_said() -> None:
    """hands-readback-ddk: it was written only at the next key press, as cut off, however long before it had been heard."""
    said = [LLMFullResponseStartFrame(), LLMTextFrame("Staging it now."), LLMFullResponseEndFrame(), *as_a_turn(TTSSpeakFrame("Draft for bananas: cherry."))]

    assert await replies(said, 2) == [Replied("Staging it now.", interrupted=False), Replied("Draft for bananas: cherry.", interrupted=False)]


async def test_a_question_hands_asks_in_the_middle_of_a_reply_is_written_between_the_two_halves_of_it() -> None:
    """The brain's stage ends the reply so far, has hands ask, and starts the reply again once the question is answered."""
    said = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Deleting the branch."),
        LLMFullResponseEndFrame(),
        *as_a_turn(TTSSpeakFrame("May it run git push?")),
        LLMFullResponseStartFrame(),
        LLMTextFrame("Done."),
        LLMFullResponseEndFrame(),
    ]

    assert await replies(said, 3) == [
        Replied("Deleting the branch.", interrupted=False),
        Replied("May it run git push?", interrupted=False),
        Replied("Done.", interrupted=False),
    ]


async def test_a_line_kept_out_of_the_context_is_no_reply_whatever_is_sent_behind_it() -> None:
    said = [*as_a_turn(TTSSpeakFrame("api: reading files.", append_to_context=False)), *as_a_turn(TTSSpeakFrame("The session api is gone."))]

    assert await replies(said, 1) == [Replied("The session api is gone.", interrupted=False)]
