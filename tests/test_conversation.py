"""The conversation as the audit log tells it: each reply written as the speaker finishes it, whoever's words it was."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from conftest import Running, running
from hands.sessions.audit import Entry, Replied
from hands.voice.conversation import record_turns, turns
from hands.voice.player import Mark, Marks
from hands.voice.spoken import FenceAggregator


class Speaker(TTSService):
    """The speaker with the synthesis taken out: started, stopped, and its text pushed by Pipecat's service as pocket-tts's are."""

    def __init__(self) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
            push_start_frame=True, push_stop_frames=True, stop_frame_timeout_s=0.05, settings=PocketTTSSettings(model=None, voice="alba", language=Language.EN)
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield TTSAudioRawFrame(b"\x00\x00", self.sample_rate, 1, context_id=context_id)


@dataclass
class Conversation:
    """The model's context and the audit log's replies, behind Pipecat's own speaker and hands' assistant side."""

    pipeline: Running
    context: LLMContext
    recorded: list[Entry] = field(default_factory=list[Entry])
    written: asyncio.Event = field(default_factory=asyncio.Event)

    async def said(self, *frames: Frame) -> None:
        """Send the frames as the model's stage sends them, and wait until the last of them has been taken."""
        taken = asyncio.Event()
        await self.pipeline.worker.queue_frames([*frames, Mark(taken.set)])
        await asyncio.wait_for(taken.wait(), 2.0)

    async def replies(self, count: int) -> list[Entry]:
        """What the audit log is told, once `count` replies are written."""
        while len(self.recorded) < count:
            self.written.clear()
            await asyncio.wait_for(self.written.wait(), 2.0)
        return self.recorded

    def ends_on(self) -> str:
        """Whose message the context ends on: what a request made from it now would end on."""
        match self.context.get_messages()[-1]:
            case {"role": str() as role}:
                return role
            case other:
                raise AssertionError(f"the context ends on a message with no role: {other!r}")


@asynccontextmanager
async def conversing() -> AsyncGenerator[Conversation]:
    context = LLMContext()
    user, assistant = turns(context, LLMUserAggregatorParams())
    async with running([LLMTextProcessor(text_aggregator=FenceAggregator()), Speaker(), assistant, Marks()]) as pipeline:
        conversation = Conversation(pipeline, context)

        def record(entry: Entry) -> None:
            conversation.recorded.append(entry)
            conversation.written.set()

        record_turns(user, assistant, record)
        yield conversation


def calling(call: str, *said_meanwhile: Frame) -> list[Frame]:
    """A reply of the model's that says it is checking and makes one call, as an API service sends it: the call is
    announced ahead of the reply's end and runs once the reply has ended, and `said_meanwhile` is said while it runs."""
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Checking."),
        FunctionCallsStartedFrame([FunctionCallFromLLM("list_sessions", call, {}, None)]),
        LLMFullResponseEndFrame(),
        FunctionCallInProgressFrame("list_sessions", call, {}, cancel_on_interruption=True),
        *said_meanwhile,
    ]


def result(call: str, run_llm: bool) -> FunctionCallResultFrame:
    return FunctionCallResultFrame("list_sessions", call, {}, {"sessions": []}, properties=FunctionCallResultProperties(run_llm=run_llm))


ANSWER = (LLMFullResponseStartFrame(), LLMTextFrame("Nothing is running."), LLMFullResponseEndFrame())
CHECKING = Replied("Checking.", interrupted=False)


async def test_a_line_hands_says_after_the_models_reply_is_written_as_heard_whole_once_it_is_said() -> None:
    """hands-readback-ddk, hands-readback-v6m: it was written only at the next key press, as cut off, however long
    before it had been heard."""
    async with conversing() as conversation:
        await conversation.said(LLMFullResponseStartFrame(), LLMTextFrame("Staging it now."), LLMFullResponseEndFrame(), TTSSpeakFrame("Draft for bananas: cherry."))

        assert await conversation.replies(2) == [Replied("Staging it now.", interrupted=False), Replied("Draft for bananas: cherry.", interrupted=False)]


async def test_a_question_hands_asks_in_the_middle_of_a_reply_is_written_between_the_two_halves_of_it() -> None:
    """The brain's stage ends the reply so far, has hands ask, and starts the reply again once the question is answered."""
    async with conversing() as conversation:
        await conversation.said(
            LLMFullResponseStartFrame(),
            LLMTextFrame("Deleting the branch."),
            LLMFullResponseEndFrame(),
            TTSSpeakFrame("May it run git push?"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Done."),
            LLMFullResponseEndFrame(),
        )

        assert await conversation.replies(3) == [
            Replied("Deleting the branch.", interrupted=False),
            Replied("May it run git push?", interrupted=False),
            Replied("Done.", interrupted=False),
        ]


async def test_a_line_kept_out_of_the_context_is_no_reply() -> None:
    async with conversing() as conversation:
        await conversation.said(TTSSpeakFrame("api: reading files.", append_to_context=False), TTSSpeakFrame("The session api is gone."))

        assert await conversation.replies(1) == [Replied("The session api is gone.", interrupted=False)]


async def test_a_line_said_while_a_call_runs_is_written_with_the_reply_that_answers_the_call() -> None:
    """Written as it is said, it would follow the call's result in the context, and the request that answers the call
    would end on an assistant message, which Claude refuses."""
    async with conversing() as conversation:
        await conversation.said(*calling("c1", TTSSpeakFrame("The session api is gone.")), result("c1", run_llm=True))
        assert conversation.ends_on() == "tool"

        await conversation.said(*ANSWER)
        assert await conversation.replies(2) == [CHECKING, Replied("The session api is gone. Nothing is running.", interrupted=False)]


async def test_a_line_said_after_a_result_the_model_is_run_on_is_written_with_the_reply_that_answers_it() -> None:
    """The model's turn is open until that reply starts: the request for it is made from this same context."""
    async with conversing() as conversation:
        await conversation.said(*calling("c1"), result("c1", run_llm=True), TTSSpeakFrame("The session api is gone."))
        assert conversation.ends_on() == "tool"

        await conversation.said(*ANSWER)
        assert await conversation.replies(2) == [CHECKING, Replied("The session api is gone. Nothing is running.", interrupted=False)]


async def test_a_line_said_while_a_call_runs_that_is_the_whole_reply_is_written_as_the_call_is_answered() -> None:
    """A playback tool's sentences: the model is not run again, so nothing later would end their turn."""
    async with conversing() as conversation:
        await conversation.said(*calling("c1", TTSSpeakFrame("That was the last of it.")), result("c1", run_llm=False))

        assert await conversation.replies(2) == [CHECKING, Replied("That was the last of it.", interrupted=False)]


async def test_a_line_said_after_a_call_that_was_the_whole_reply_is_written_once_it_is_said() -> None:
    async with conversing() as conversation:
        await conversation.said(*calling("c1"), result("c1", run_llm=False), TTSSpeakFrame("The session api is gone."))

        assert await conversation.replies(2) == [CHECKING, Replied("The session api is gone.", interrupted=False)]
