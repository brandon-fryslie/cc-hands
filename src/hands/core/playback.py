"""Where playback is: the reading on the speaker, how far it has played, and where earlier readings were cut off.

The user cuts in often, to ask which file or to answer something else, so every interrupted reading must be resumable.
Where it stopped lives here, in the daemon, never in the model's memory: a model asked to go back paraphrases what it
remembers, and the user hears a different reading than the one they cut off.

A reading is the run of sentences handed to the speaker since it last fell quiet: a reply, a line said as written, or
both back to back. A position is what the TTS service hands the speaker as one: a sentence of a reply, or a line said as
written, whole. pocket-tts reports no word timings, so there is nothing finer, and "where it stopped" is the one that was
playing, said again from its start.
"""

from dataclasses import dataclass, replace

# How many cut-off readings are kept to go back to. Each barge-in that is never resumed leaves one, so the stack is
# bounded by the user's own cutting in, and past this the oldest goes: nobody goes back nine readings by voice.
BOOKMARKS = 8


@dataclass(frozen=True)
class Bookmark:
    """A reading that was cut off, and the sentence that was playing when it was."""

    sentences: tuple[str, ...]
    at: int


@dataclass(frozen=True)
class Playback:
    """The latest reading, how many of its sentences have finished playing, the sentences of it hands has yet to hand
    the speaker, and the readings cut off.

    A reading every sentence of which has played is over, and the next sentence handed starts a new one. `pressed_on`
    is the reading the user's latest barge-in found, cut off or played to its end: what "say that again" says, whatever
    the model said before asking for it. `stopped` is that reading when the barge-in cut it off, kept apart from the
    earlier ones because going back is going back from it, to the reading before.
    """

    reading: tuple[str, ...] = ()
    played: int = 0
    coming: tuple[str, ...] = ()
    pressed_on: tuple[str, ...] = ()
    stopped: Bookmark | None = None
    interrupted: tuple[Bookmark, ...] = ()

    @property
    def over(self) -> bool:
        return self.played == len(self.reading) and not self.coming

    @property
    def bookmarks(self) -> tuple[Bookmark, ...]:
        """Every reading cut off, most recent last."""
        return (*self.interrupted, *_held(self.stopped))


def begun(playback: Playback) -> Playback:
    """A new reading starts; the reading the latest barge-in cut off joins the earlier ones."""
    return Playback(pressed_on=playback.pressed_on, interrupted=playback.bookmarks[-BOOKMARKS:])


def queued(playback: Playback, sentences: tuple[str, ...]) -> Playback:
    """Hands will hand the speaker these next, and knows them before it does: the rest of the reading playing, or a new
    one, as for each sentence handed."""
    start = begun(playback) if playback.over else playback
    return replace(start, coming=(*start.coming, *sentences))


def handed(playback: Playback, sentence: str) -> Playback:
    """A sentence is on its way to the speaker: the next of the reading playing, or the first of a new one. It is the
    next line hands queued when it is that line; anything else handed meanwhile, a word of the model's or a line
    another part of hands says, takes its own place in the reading ahead of the lines still coming."""
    start = begun(playback) if playback.over else playback
    coming = start.coming[1:] if start.coming[:1] == (sentence,) else start.coming
    return replace(start, reading=(*start.reading, sentence), coming=coming)


def finished(playback: Playback) -> Playback:
    """The next sentence of the reading has played to its end. The speaker plays them in the order they were handed, so
    the one that finished is the first that had not; one reported after its reading was cut off changes nothing."""
    return replace(playback, played=min(playback.played + 1, len(playback.reading)))


def cut(playback: Playback) -> Playback:
    """The user barged in: the reading stops, with what hands had yet to hand of it, and the sentence it stopped on is
    bookmarked to go back to. A reading that had played to its end is not cut off, and leaves nothing to go back to."""
    whole = (*playback.reading, *playback.coming)
    pressed = replace(playback, pressed_on=whole)
    return pressed if playback.over else replace(pressed, reading=whole, played=len(whole), coming=(), stopped=Bookmark(whole, playback.played))


@dataclass(frozen=True)
class Replay:
    """Sentences to say again, in order, as they were said."""

    sentences: tuple[str, ...]


@dataclass(frozen=True)
class NothingCut:
    """No reading was cut off, so there is nowhere to go back to."""


@dataclass(frozen=True)
class NothingSaid:
    """Nothing has been said yet, so there is nothing to say again."""


@dataclass(frozen=True)
class LastOne:
    """The sentence skipped was the last of its reading, so nothing of it is left to say."""


Played = Replay | NothingCut | NothingSaid | LastOne


def resume(playback: Playback) -> tuple[Playback, Played]:
    """"Go back to what you were talking about": the latest reading cut off before the one the user just cut in on, from
    the start of the sentence it stopped on. The one cut in on is what they are going back from, and is left; with none
    before it, it is the one gone back to."""
    return _popped(playback, playback.interrupted or _held(playback.stopped), 0)


def skip(playback: Playback) -> tuple[Playback, Played]:
    """"Skip that": the latest reading cut off, from the sentence after the one it stopped on."""
    return _popped(playback, playback.bookmarks, 1)


def repeat(playback: Playback) -> tuple[Playback, Played]:
    """"Say that again": the reading the user's latest barge-in found, whole, whether it was cut off or played to its end.
    Said whole, it no longer waits to be gone back to."""
    return replace(playback, stopped=None), Replay(playback.pressed_on) if playback.pressed_on else NothingSaid()


def _popped(playback: Playback, stack: tuple[Bookmark, ...], past: int) -> tuple[Playback, Played]:
    """The latest bookmark of `stack` taken off, the rest left as the readings cut off, and its reading from `past`
    sentences after the one it stopped on."""
    if not stack:
        return playback, NothingCut()
    *earlier, latest = stack
    rest = latest.sentences[latest.at + past :]
    return replace(playback, stopped=None, interrupted=tuple(earlier)), Replay(rest) if rest else LastOne()


def _held(bookmark: Bookmark | None) -> tuple[Bookmark, ...]:
    return () if bookmark is None else (bookmark,)
