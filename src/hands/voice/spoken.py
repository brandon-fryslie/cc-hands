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

One path that crosses this seam does not want a summary's liberties. A draft readback (`voice/readback.py`)
is read out so the user can check what will be sent, and a path in it is spoken as its file name like any
other — "fix src/auth.py" is heard as "fix auth". That is not a regression this filter introduced, because
the unfiltered string was `s-r-c slash auth dot p y` and could not be checked by ear either, and the
readback already passes through the model before it is spoken. But it does mean a readback can no longer
be relied on to distinguish two files whose names agree, which is tracked as its own ticket rather than
solved by giving this function a mode [LAW:no-mode-explosion].

The filter is stateless, so a barge-in in the middle of a sentence leaves nothing in it to reset.
"""

from collections.abc import AsyncIterator

from loguru import logger
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from hands.core.spoken import open_fence, spoken


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

    A fence opens and closes only at the end of a line, so that is the only place the block is looked for,
    and whether one is open is read off the buffer by the same rules `spoken` reads it by
    [LAW:one-source-of-truth]. What came before the opening fence goes on ahead rather than waiting out the
    block. A block the reply never closes is what `flush` hands on when the reply ends, and `spoken` says
    an unclosed block as the block it is.
    """

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (untyped in Pipecat)
        # Whether the buffer begins with a block not yet closed: recorded at the line end that opened it,
        # so the characters inside it are not offered to the sentence splitter one by one.
        self._in_block = False

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for char in text:
            self._text += char
            if char == "\n":
                piece = self._line_ended()
                if piece is not None:
                    yield piece
                    continue
            if not self._in_block:
                sentence = await self._check_sentence_with_lookahead(char)
                if sentence is not None:
                    yield sentence

    def _line_ended(self) -> Aggregation | None:
        """What a finished line lets go of: the text ahead of a block it opened, or the block it closed."""
        opened = open_fence(self._text)
        if not self._in_block and opened is not None:
            self._in_block, self._needs_lookahead = True, False
            ahead, self._text = self._text[:opened], self._text[opened:]
            return Aggregation(text=ahead.strip(" "), type=AggregationType.SENTENCE)
        if self._in_block and opened is None:
            self._in_block = False
            block, self._text = self._text, ""
            return Aggregation(text=block.strip(" "), type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self) -> None:
        await super().handle_interruption()
        self._in_block = False

    async def reset(self) -> None:
        await super().reset()
        self._in_block = False
