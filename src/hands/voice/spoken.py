"""The one filter every utterance crosses on its way to the speaker.

Pipecat applies a TTS service's text filters to everything it is about to synthesise — the text of a
`TTSSpeakFrame`, and each aggregated sentence of a model's streamed reply alike. That is the seam this
ticket asks for and the reason nothing else here has to be trusted: the narrator, the announcements, the
system's own reports and the intermediary's own words all pass through this one function, and text that
never went through it cannot reach the speaker at all [LAW:single-enforcer].

What arrives here is a whole utterance for a `TTSSpeakFrame` and one aggregated piece at a time for a
streamed reply. A list the intermediary streams is therefore seen an item at a time and is not counted
aloud as a sequence, where the same list inside a summary is.

A fenced block is the one shape that cannot survive being split: its continuation arrives carrying no
fence and is read out as the ordinary text it then resembles. So the reply is not split inside one.
`FenceAggregator` breaks the stream into sentences everywhere else and holds a block whole until its
closing fence, and it lives in front of the TTS service in Pipecat's `LLMTextProcessor`, which flushes it
when a reply ends and resets it when the user interrupts. That is where the open fence belongs. Carried
in this filter instead, it was tried and reverted: Pipecat never tells a text filter that a reply ended,
so a reply that ended inside a fence left every later utterance replaced by a block announcement
[LAW:no-ambient-temporal-coupling]. The aggregator's lifetime is the reply's, by Pipecat's own frames.

The filtered text is also what the intermediary remembers: Pipecat builds the frame it appends to the
assistant context from the text a filter returned. Of the five places a `TTSSpeakFrame` is built, three
keep their text — `narrator.py` twice and `speech.py` once — and both of those files say in their own
words why: the context is kept so the model can answer about *what the user heard*. Until this filter
existed it held what was sent to the speaker instead, which was never the same string. What it costs is
that the model cannot read an exact path or sha back out of its own memory; it has the session tools for
facts, and what it said is now what was heard.

One path that crosses this seam wants every character heard: a draft readback (`voice/readback.py`), read
out so the user can check what will be typed. It does not get a mode here [LAW:no-mode-explosion]. The
readback says its text and resolutions with `spelled`, whose output carries no tell for this filter to act
on, and its line breaks as words, so no rule here that reads the shape of a line reaches inside a draft.

The filter is stateless, so a barge-in in the middle of a sentence leaves nothing in it to reset.
"""

from collections.abc import AsyncIterator

from loguru import logger
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from hands.core.spoken import fence_after, spoken


class SpokenForm(BaseTextFilter):
    """Puts everything on its way to speech into a form that can be heard, and says what it had to drop."""

    async def filter(self, text: str) -> str:
        said = spoken(text)
        for leak in said.leaks:
            # [LAW:no-silent-failure] the user hears that something was there, and the log says what it
            # was: a leak means something upstream handed the ear what it owed the summariser, and the
            # fault is there rather than here. Said either way — never read out, never silently dropped.
            logger.warning(f"{leak} reached the speaker, so it was said as what it was rather than read out")
        return said.text


class FenceAggregator(SimpleTextAggregator):
    """A streamed reply in sentences, except that a fenced block is one piece from its opening fence to its close.

    A fence opens and closes only on a line of its own, so each line is read as it finishes, by the same step
    `spoken` reads a block by [LAW:one-source-of-truth] — the line, not the buffer, which a sentence break
    may have started partway through one. What came before the opening fence goes on ahead rather than
    waiting out the block. A block the reply never closes is what `flush` hands on when the reply ends, and
    `spoken` says an unclosed block as the block it is.
    """

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        self._fence: str | None = None
        self._line = ""

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for char in text:
            self._text += char
            self._line += char
            piece = self._line_ended() if char == "\n" else None
            if piece is None and self._fence is None:
                piece = await self._check_sentence_with_lookahead(char)
            # Nothing ahead of a fence that opens the reply: no piece, rather than an empty one.
            if piece is not None and piece.text.strip():
                yield piece

    def _line_ended(self) -> Aggregation | None:
        """What a finished line lets go of: the text ahead of a block it opened, or the block it closed."""
        line, self._line = self._line, ""
        was, self._fence = self._fence, fence_after(line.rstrip("\n"), self._fence)
        match was, self._fence:
            case None, str():
                self._needs_lookahead = False
                # Where the opening line began, unless a sentence already went on ahead with the start of it.
                begins = max(0, len(self._text) - len(line))
                ahead, self._text = self._text[:begins], self._text[begins:]
                return Aggregation(text=ahead.strip(" "), type=AggregationType.SENTENCE)
            case str(), None:
                block, self._text = self._text, ""
                return Aggregation(text=block.strip(" "), type=AggregationType.SENTENCE)
            case _:
                return None

    async def handle_interruption(self) -> None:
        await super().handle_interruption()
        self._fence, self._line = None, ""

    async def reset(self) -> None:
        await super().reset()
        self._fence, self._line = None, ""


class EndsReplies(TTSService):
    """A TTS service for which the model's reply is over at its end frame, so a line hands says after it is a turn of its own.

    Pipecat's service ends the assistant turn of a `TTSSpeakFrame` kept in the context only while no reply is under
    way, and one that pushes its own text frames, as pocket-tts does, takes a reply as under way from its start frame
    until the next barge-in (still so upstream on 2026-10-04). Every kept line said after a reply then stayed an open
    turn, written to the context and the audit log at the next key press, as cut off.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case LLMFullResponseEndFrame():
                # [LAW:one-source-of-truth] the end frame is what says the reply ended; the service's own note of it follows.
                self._llm_response_started = False
            case _:
                pass
