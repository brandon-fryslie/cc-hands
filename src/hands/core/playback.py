"""Where playback is: the reading on the speaker, how far it has played, and where earlier readings were cut off.

The user cuts in often, to ask which file or to answer something else, so every interrupted reading must be resumable.
Where it stopped lives here, in the daemon, never in the model's memory: a model asked to go back paraphrases what it
remembers, and the user hears a different reading than the one they cut off.

A reading is the run of sentences handed to the speaker since it last fell quiet: a reply, a line said as written, or
both back to back. A sentence is the finest position there is, since pocket-tts reports no word timings, so "where it
stopped" is the sentence that was playing, said again from its start.
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
    """The latest reading, how many of its sentences have finished playing, and the readings cut off, most recent last.

    A reading every sentence of which has played is over, and the next sentence handed starts a new one; it stays
    held until then, as what "say that again" says.
    """

    reading: tuple[str, ...] = ()
    played: int = 0
    interrupted: tuple[Bookmark, ...] = ()

    @property
    def over(self) -> bool:
        return self.played == len(self.reading)


def handed(playback: Playback, sentence: str) -> Playback:
    """A sentence is on its way to the speaker: the next of the reading playing, or the first of a new one."""
    reading = (sentence,) if playback.over else (*playback.reading, sentence)
    return replace(playback, reading=reading, played=0 if playback.over else playback.played)


def finished(playback: Playback) -> Playback:
    """The next sentence of the reading has played to its end. The speaker plays them in the order they were handed, so
    the one that finished is the first that had not; one reported after its reading was cut off changes nothing."""
    return replace(playback, played=min(playback.played + 1, len(playback.reading)))


def cut(playback: Playback) -> Playback:
    """The user barged in: the reading stops, and the sentence it stopped on is bookmarked to go back to. A reading that
    had played to its end is not cut off, and leaves nothing to go back to."""
    left = () if playback.over else (Bookmark(playback.reading, playback.played),)
    return replace(playback, played=len(playback.reading), interrupted=(*playback.interrupted, *left)[-BOOKMARKS:])


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
    """"Go back to what you were talking about": the latest reading cut off, from the start of the sentence it stopped on."""
    return _popped(playback, 0)


def skip(playback: Playback) -> tuple[Playback, Played]:
    """"Skip that": the latest reading cut off, from the sentence after the one it stopped on."""
    return _popped(playback, 1)


def repeat(playback: Playback) -> tuple[Playback, Played]:
    """"Say that again": the latest reading, whole, whether it was cut off or played to its end."""
    return playback, Replay(playback.reading) if playback.reading else NothingSaid()


def _popped(playback: Playback, past: int) -> tuple[Playback, Played]:
    """The latest bookmark taken off the stack, and its reading from `past` sentences after the one it stopped on."""
    if not playback.interrupted:
        return playback, NothingCut()
    *earlier, latest = playback.interrupted
    rest = latest.sentences[latest.at + past :]
    return replace(playback, interrupted=tuple(earlier)), Replay(rest) if rest else LastOne()
