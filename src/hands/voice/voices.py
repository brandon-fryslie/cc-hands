"""The voice hands speaks in: one of Pocket TTS's own, chosen by the user by ear, and kept in the home so it outlives
the daemon and needs no restart.

[LAW:one-source-of-truth] the file is the choice. The daemon builds its TTS service in the voice the file names, and
each change goes to the file and to the service together; nothing else holds a copy.
"""

import asyncio
from collections.abc import Callable
from typing import NewType

from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.pocket_tts.tts import PocketTTSSettings, language_to_pocket_tts_language
from pipecat.transcriptions.language import Language
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # pyright: ignore[reportPrivateUsage]
from pocket_tts.utils.utils import download_if_necessary, get_predefined_voice

from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

Voice = NewType("Voice", str)

# [LAW:one-source-of-truth] the voices on offer are the ones the installed pocket_tts resolves by name itself, each to
# a state primed by the very weights it loads; a state primed elsewhere is a cache from another network, and the loader
# cannot tell.
VOICES: tuple[Voice, ...] = tuple(Voice(name) for name in _ORIGINS_OF_PREDEFINED_VOICES)

# The voice hands speaks in until the user chooses another (the user's choice of default, 2026-10-02).
DEFAULT = Voice("charles")

# The language the TTS service speaks, which picks the set of embeddings a voice's name resolves to.
LANGUAGE = Language.EN

# Whatever a voice needs before it can be spoken in, made ready, or the reason it cannot be.
Fetch = Callable[[Voice], None]


def parse_voice(name: str) -> Voice:
    """The voice a name means, read as a person says it: "Bill Boerst" is bill_boerst."""
    # [LAW:parse-dont-validate] the one crossing from a spoken or written name; past it, a Voice is one pocket_tts has.
    said = Voice("_".join(name.lower().split()))
    if said not in VOICES:
        raise Rejected(f"hands has no voice called {name!r}; it has {', '.join(VOICES)}")
    return said


def spoken(voice: Voice) -> str:
    """The voice's name as it is said aloud."""
    return voice.replace("_", " ").title()


def chosen(home: Home) -> Voice:
    """The voice the user chose; the default where they never chose one."""
    try:
        # Bytes, not text: a file edited by hand can hold anything, and whatever names no voice is refused alike.
        written = home.voice.read_bytes().strip()
    except FileNotFoundError:
        return DEFAULT
    try:
        return parse_voice(written.decode())
    except (UnicodeDecodeError, Rejected) as error:
        raise Rejected(f"{home.voice} says {written!r}, which names no voice hands has; choose another, or remove the file to speak in {spoken(DEFAULT)}") from error


def keep(home: Home, voice: Voice) -> None:
    # [LAW:no-ambient-temporal-coupling] replaced whole, so a reader never sees half a name.
    replace_whole(home.voice, f"{voice}\n", 0o644)


def fetched(voice: Voice) -> None:
    """The voice's embedding in the local cache, downloaded the first time, as the TTS service resolves the name."""
    download_if_necessary(get_predefined_voice(language=language_to_pocket_tts_language(LANGUAGE), name=voice))


def sample(voice: Voice) -> str:
    """What a voice says when the user hears it: its name, and a line of hands' own that reports nothing, so said again
    after a barge-in it cannot pass for a session's news."""
    return f"This is {spoken(voice)}. This is how I would tell you a session has finished its turn and is waiting for you."


class Voices:
    """The voices hands can speak in, heard and chosen through `lines`, the processor standing ahead of the TTS service.

    A change of voice is a frame in order with the speech around it, so what was handed to the speaker before it is
    said in the old voice and what comes after in the new. Pipecat keeps a settings frame through a barge-in, which
    drops only the speech, so a hearing cut short still ends back in the chosen voice.
    """

    def __init__(self, home: Home, lines: FrameProcessor, fetch: Fetch) -> None:
        self._home = home
        self._lines = lines
        self._fetch = fetch
        # [LAW:no-ambient-temporal-coupling] the one owner of a change of voice: Pipecat runs a reply's calls side by
        # side, and a hearing and a choice interleaved would leave a sample in another's voice, or the speaker in a
        # voice the file does not hold. Each runs whole, one after another.
        self._changing = asyncio.Lock()

    async def speaking_in(self) -> Voice:
        return await asyncio.to_thread(chosen, self._home)

    async def hear(self, voices: tuple[Voice, ...]) -> Voice:
        """Each voice says its sample, in order, and then hands goes back to the voice it speaks in, which it returns.
        A voice that cannot be fetched is refused before any is heard."""
        async with self._changing:
            now = await self.speaking_in()
            for voice in voices:
                await asyncio.to_thread(self._fetch, voice)
            try:
                for voice in voices:
                    await self._speak_in(voice)
                    await self._lines.push_frame(TTSSpeakFrame(sample(voice), append_to_context=False))
            finally:
                # A hearing cancelled or failed partway still hands the speaker back the voice it speaks in.
                await self._speak_in(now)
            return now

    async def use(self, voice: Voice) -> None:
        """From the next thing hands says on, and across restarts, it speaks in `voice`; a voice that cannot be fetched
        is refused before it is kept, so hands is never left in one it cannot speak."""
        async with self._changing:
            await asyncio.to_thread(self._fetch, voice)
            await asyncio.to_thread(keep, self._home, voice)
            await self._speak_in(voice)

    async def _speak_in(self, voice: Voice) -> None:
        await self._lines.push_frame(TTSUpdateSettingsFrame(delta=PocketTTSSettings(voice=voice)))
