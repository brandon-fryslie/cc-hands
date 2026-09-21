"""The one filter every utterance crosses on its way to the speaker.

Pipecat applies a TTS service's text filters to everything it is about to synthesise — the text of a
`TTSSpeakFrame`, and each aggregated sentence of a model's streamed reply alike. That is the seam this
ticket asks for and the reason nothing else here has to be trusted: the narrator, the announcements, the
system's own reports and the intermediary's own words all pass through this one function, and text that
never went through it cannot reach the speaker at all [LAW:single-enforcer].

What arrives here is a whole utterance for a `TTSSpeakFrame` and one aggregated sentence at a time for a
streamed reply. So this filter remembers exactly one thing between calls: the fence a chunk ended inside.
A fenced block streamed a sentence at a time opens in one chunk and continues in the next, and a chunk
carrying no fence of its own was read out as the ordinary text it then resembled — the block announced
once and spoken anyway, which is the one thing this module exists to prevent.

Holding that much is safe because Pipecat asks a filter to let it go: `handle_interruption` is called on
every filter when an interruption frame arrives, and `reset_interruption` before each pass. A barge-in
mid-block therefore clears the carry rather than leaving the next reply suppressed behind a fence nobody
closed [LAW:no-ambient-temporal-coupling]. Nothing else is remembered, and the carry is a value `spoken`
returns rather than state it keeps, so the parsing stays in one pure place [LAW:one-source-of-truth].

A list the intermediary streams is still seen an item at a time and so is not counted aloud as a sequence,
where the same list inside a summary is. That one needs the whole list in one piece and cannot be had a
sentence at a time; every rule that makes text sayable at all applies to each chunk regardless.

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

Stateless, so a barge-in in the middle of a sentence leaves nothing to reset. In a system where the user
interrupts constantly, a filter holding half an utterance between calls is a bug waiting for the second
half that never comes [LAW:no-ambient-temporal-coupling].
"""

from loguru import logger
from pipecat.utils.text.base_text_filter import BaseTextFilter

from hands.core.spoken import Fence, spoken


class SpokenForm(BaseTextFilter):
    """Puts everything on its way to speech into a form that can be heard, and says what it had to drop."""

    def __init__(self) -> None:
        # The one thing carried between calls: the fence a chunk ended inside, so the next chunk of the
        # same reply is known to be inside it too. Cleared on interruption, below.
        self._inside: Fence | None = None

    async def filter(self, text: str) -> str:
        said = spoken(text, self._inside)
        self._inside = said.unclosed
        for leak in said.leaks:
            # [LAW:no-silent-failure] the user hears that something was there, and the log says what it
            # was: a leak means something upstream handed the ear what it owed the summariser, and the
            # fault is there rather than here. Said either way — never read out, never silently dropped.
            logger.warning(f"{leak} reached the speaker, so it was said as what it was rather than read out")
        return said.text

    async def handle_interruption(self) -> None:
        """A reply abandoned mid-block takes its open fence with it.

        Kept, the fence would swallow the beginning of whatever the user asked for next — the listener
        would hear a block of code announced in place of the answer to a new question.
        """
        self._inside = None
