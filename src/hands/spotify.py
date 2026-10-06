"""Spotify on this Mac: its app, played through the AppleScript it answers, and its catalogue, searched through the Web API.

The app needs nothing set up: macOS asks the user once whether hands may control Spotify. The catalogue needs a Spotify
developer app's client ID and secret in hands' environment, SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET, since the
AppleScript Spotify answers plays what is named by its link but finds nothing by name.
"""

import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import aiohttp

from hands.sessions.child import Ran, run
from hands.sessions.wide import annotate

# osascript answers in a fraction of a second once Spotify is up; starting Spotify to play takes some seconds.
TELL_SECONDS = 20.0
SEARCH_SECONDS = 10.0
TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
SEARCH_LIMIT = 5
CREDENTIALS = ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")

Kind = Literal["track", "album", "artist", "playlist"]
State = Literal["playing", "paused", "stopped", "not running"]

# One field from the next in what a script prints: a character no track name holds.
_FIELD = "\x1f"
_GUARD = 'if application "Spotify" is not running then return "not running"'
# What osascript says when macOS has not let hands send Spotify Apple events: errAEEventNotPermitted.
_NOT_PERMITTED = "(-1743)"

# What Spotify's AppleScript plays: a track, or the album, artist, playlist, show, or episode it plays from the top. The
# link the app's Share menu copies is the same item on the web, with an optional locale and a query of its own.
_URI = re.compile(r"spotify:(track|album|artist|playlist|show|episode):([0-9A-Za-z]{22})")
_URL = re.compile(r"https://open\.spotify\.com/(?:intl-[a-z]{2}(?:-[A-Za-z]{2})?/)?(track|album|artist|playlist|show|episode)/([0-9A-Za-z]{22})(?:\?\S*)?")


def link(text: str) -> str | None:
    """The spotify: URI `text` names, a URI or the open.spotify.com link of one; None for anything else."""
    stripped = text.strip()
    match = _URI.fullmatch(stripped) or _URL.fullmatch(stripped)
    return None if match is None else f"spotify:{match[1]}:{match[2]}"


class Refused(Exception):
    """Spotify, or macOS on its behalf, did not do what it was told; the message says why in words for the user."""


type Tell = Callable[[Sequence[str], Sequence[str]], Awaitable[Ran]]


async def osascript(script: Sequence[str], argv: Sequence[str]) -> Ran:
    # Every value goes in as an argument, never spliced into the script, so nothing in it needs quoting.
    return await run("osascript", *(part for line in script for part in ("-e", line)), "--", *argv, timeout=TELL_SECONDS)


@dataclass(frozen=True)
class Playing:
    """What Spotify is playing: its state, and the track in it, which a stopped or closed Spotify has none of."""

    state: State
    track: str | None = None
    artist: str | None = None
    album: str | None = None
    link: str | None = None
    position_seconds: int | None = None
    duration_seconds: int | None = None


class Player:
    """The Spotify app, told through AppleScript. Only `play` starts it: everything else, asked of a Spotify that is not
    running, says so and starts nothing."""

    def __init__(self, tell: Tell = osascript) -> None:
        self._tell = tell

    async def now_playing(self) -> Playing:
        printed = await self._run(
            [
                "on run argv",
                _GUARD,
                'tell application "Spotify"',
                "if player state is stopped then return \"stopped\"",
                "set t to current track",
                # Spotify's AppleScript gives a track's duration in milliseconds and the position in seconds (1.2.x).
                f'return (player state as text) & (character id 31) & my field(name of t) & (character id 31) & my field(artist of t) & (character id 31) & my field(album of t) & (character id 31) & (spotify url of t) & (character id 31) & ((player position) as integer) & (character id 31) & ((duration of t) div 1000)',
                "end tell",
                "end run",
                # An episode or an ad has no artist or album: `missing value`, which `&` would print as those words.
                "on field(v)",
                'if v is missing value then return ""',
                "return v as text",
                "end field",
            ],
            [],
        )
        match printed.split(_FIELD):
            case ["not running"]:
                return Playing("not running")
            case ["stopped"]:
                return Playing("stopped")
            case [("playing" | "paused") as state, track, artist, album, uri, position, duration]:
                return Playing(cast(State, state), track or None, artist or None, album or None, uri, int(position), int(duration))
            case _:
                raise Refused(f"Spotify answered what is playing with {printed!r}, which hands cannot read")

    async def play(self, uri: str | None) -> None:
        """Play `uri` from its start, or go on with what was playing; starts Spotify where it is not running."""
        if uri is None:
            await self._run(["on run argv", 'tell application "Spotify" to play', "end run"], [])
        else:
            await self._run(["on run argv", 'tell application "Spotify" to play track (item 1 of argv)', "end run"], [uri])

    async def pause(self) -> None:
        await self._guarded("pause")

    async def skip(self, to: Literal["next", "previous"]) -> None:
        await self._guarded(f"{to} track")

    async def volume(self, level: int) -> None:
        await self._guarded("set sound volume to (item 1 of argv) as integer", str(level))

    async def shuffle(self, on: bool) -> None:
        await self._guarded(f"set shuffling to {str(on).lower()}")

    async def repeat(self, on: bool) -> None:
        await self._guarded(f"set repeating to {str(on).lower()}")

    async def _guarded(self, command: str, *argv: str) -> None:
        """Run `command` in a Spotify that is running; refused where it is not, rather than starting it to obey."""
        if await self._run(["on run argv", _GUARD, 'tell application "Spotify"', command, "end tell", 'return "done"', "end run"], argv) == "not running":
            raise Refused("Spotify is not running")

    async def _run(self, script: Sequence[str], argv: Sequence[str]) -> str:
        try:
            ran = await self._tell(script, argv)
        except TimeoutError as error:
            raise Refused(f"Spotify did not answer in {TELL_SECONDS:.0f}s") from error
        except OSError as error:
            raise Refused(f"osascript could not start: {error}") from error
        if ran.returncode != 0:
            err = ran.err.decode().strip()
            if _NOT_PERMITTED in err:
                raise Refused("macOS has not let hands control Spotify: allow it under System Settings, Privacy & Security, Automation")
            raise Refused(f"Spotify refused: {err}")
        return ran.out.decode().strip()


@dataclass(frozen=True)
class Found:
    """One item of Spotify's catalogue: its name, who made it where it has a maker, and the link that plays it."""

    name: str
    by: str | None
    link: str


def said(fact: Playing | Found) -> Mapping[str, object]:
    """`fact` as a tool's result: its fields, leaving out those it has none of."""
    return {name: value for name, value in vars(fact).items() if value is not None}


@dataclass(frozen=True)
class Credentials:
    client_id: str
    client_secret: str


@dataclass(frozen=True)
class Missing:
    """No credentials: `why` says which are missing and where to get them."""

    why: str


def credentials(environment: Mapping[str, str]) -> Credentials | Missing:
    """The catalogue's credentials, from the environment hands was started in."""
    client_id, client_secret = (environment.get(name, "").strip() for name in CREDENTIALS)
    if client_id and client_secret:
        return Credentials(client_id, client_secret)
    missing = " and ".join(name for name in CREDENTIALS if not environment.get(name, "").strip())
    return Missing(f"searching Spotify needs {missing} in hands' environment: a Spotify developer app's, made at https://developer.spotify.com/dashboard")


class Catalogue:
    """Spotify's catalogue, searched as a developer app: a client-credentials token, fetched when there is none or it
    has run out, and held until then."""

    def __init__(self, held: Credentials | Missing, clock: Callable[[], float] = time.monotonic, token_url: str = TOKEN_URL, search_url: str = SEARCH_URL) -> None:
        self._held = held
        self._clock = clock
        self._token_url = token_url
        self._search_url = search_url
        # [LAW:no-shared-mutable-globals] owned here: the token and the clock time it runs out at, written by `_token`.
        self._token: tuple[str, float] | None = None

    async def search(self, query: str, kind: Kind) -> list[Found]:
        match self._held:
            case Missing(why=why):
                raise Refused(why)
            case Credentials() as held:
                pass
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=SEARCH_SECONDS)) as http:
                token = await self._token_for(http, held)
                async with http.get(self._search_url, params={"q": query, "type": kind, "limit": str(SEARCH_LIMIT)}, headers={"Authorization": f"Bearer {token}"}) as response:
                    if response.status != 200:
                        raise Refused(f"Spotify's search answered {response.status}: {(await response.text()).strip()}")
                    answered = await response.json()
        except (aiohttp.ClientError, TimeoutError) as error:
            raise Refused(f"Spotify's search could not be reached: {error!r}") from error
        # Spotify's search lists a playlist it cannot show as a null in place of it.
        return [_found(kind, item) for item in answered[f"{kind}s"]["items"] if item is not None]

    async def _token_for(self, http: aiohttp.ClientSession, held: Credentials) -> str:
        now = self._clock()
        if self._token is not None and now < self._token[1]:
            annotate(spotify_token="held")
            return self._token[0]
        async with http.post(self._token_url, data={"grant_type": "client_credentials"}, headers={"Authorization": aiohttp.BasicAuth(held.client_id, held.client_secret).encode()}) as response:
            if response.status != 200:
                raise Refused(f"Spotify refused hands' developer app credentials ({response.status}): {(await response.text()).strip()}")
            granted = await response.json()
        annotate(spotify_token="fetched")
        # Fetched again a minute early, so a search never goes out on a token that runs out on the way.
        self._token = (cast(str, granted["access_token"]), now + float(granted["expires_in"]) - 60.0)
        return self._token[0]


def _found(kind: Kind, item: Mapping[str, object]) -> Found:
    match kind:
        case "track" | "album":
            artists = cast(list[Mapping[str, object]], item["artists"])
            return Found(cast(str, item["name"]), ", ".join(cast(str, artist["name"]) for artist in artists), cast(str, item["uri"]))
        case "artist":
            return Found(cast(str, item["name"]), None, cast(str, item["uri"]))
        case "playlist":
            owner = cast(Mapping[str, object], item["owner"])
            return Found(cast(str, item["name"]), cast(str | None, owner.get("display_name")), cast(str, item["uri"]))
