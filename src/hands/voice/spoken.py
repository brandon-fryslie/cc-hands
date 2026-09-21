"""The one filter every utterance crosses on its way to the speaker.

Pipecat applies a TTS service's text filters to everything it is about to synthesise — the text of a
`TTSSpeakFrame`, and each aggregated sentence of a model's streamed reply alike. That is the seam this
ticket asks for and the reason nothing else here has to be trusted: the narrator, the announcements, the
system's own reports and the intermediary's own words all pass through this one function, and text that
never went through it cannot reach the speaker at all [LAW:single-enforcer].

What arrives here is a whole utterance for a `TTSSpeakFrame` and one aggregated sentence at a time for a
streamed reply, and two things follow from the second. A list the intermediary streams is seen an item at a
time and is not counted aloud as a sequence, where the same list inside a summary is. And a block that
spans chunks is only seen in the chunk its fence lands in: the rest arrives carrying no fence and is read
out as the ordinary text it now looks like. Neither can be fixed by holding state here — a filter keeping
half an utterance across calls is a bug waiting for the barge-in that never sends the second half — so the
block that spans chunks is a limit of this seam and not of `spoken`, and closing it means skipping the
block at the aggregator, before it is ever broken up. Every rule that makes text sayable at all still
applies to every chunk, which is the guarantee that does hold here.

The filtered text is also what the intermediary remembers: Pipecat builds the frame it appends to the
assistant context from the text a filter returned. That is the intent the two `TTSSpeakFrame` sites state
in their own words — the context is kept so the model can answer about *what the user heard* — and until
this filter existed it held what was sent to the speaker instead, which was never the same string. What it
costs is that the model cannot read an exact path or sha back out of its own memory; it has the session
tools for facts, and what it said is what was heard.

Stateless, so a barge-in in the middle of a sentence leaves nothing to reset. In a system where the user
interrupts constantly, a filter holding half an utterance between calls is a bug waiting for the second
half that never comes [LAW:no-ambient-temporal-coupling].
"""

from loguru import logger
from pipecat.utils.text.base_text_filter import BaseTextFilter

from hands.core.spoken import spoken


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
