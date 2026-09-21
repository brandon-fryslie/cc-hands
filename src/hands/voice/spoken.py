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
out as the ordinary text it now looks like, which is tracked as hands-narration-2mc.1zu.

Carrying the open fence between calls was tried and reverted, and the reason is worth keeping. Pipecat
does support a stateful filter — it calls `handle_interruption` on every filter when an interruption
frame arrives — but it never tells a text filter that a reply *ended*: `LLMFullResponseEndFrame` and
`EndFrame` are handled by the service without reaching the filters. So a carry has no bounded lifetime.
A reply that legitimately ends inside a fence — truncated output, a model that forgets the closer —
leaves it set, and every later utterance, announcements included, is replaced by a block announcement
until the user happens to interrupt. That mutes the assistant, which is worse than the fault it fixes
[LAW:no-ambient-temporal-coupling]. Closing this properly means skipping the block where it is broken
up, at the aggregator, and not here.

Every rule that makes text sayable at all still applies to every chunk, which is the guarantee that does
hold here.

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
