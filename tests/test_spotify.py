from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from hands.sessions.child import Ran
from hands.sessions.wide import WideEvent
from hands.spotify import Catalogue, Credentials, Missing, Player, Refused, credentials, link
from hands.voice.tool import Body
from hands.voice.tools import audited, spotify_tools

TRACK = "4uLU6hMCjMI75M1A2tKUQC"
PLAYLIST = "37i9dQZF1DXcBWIGoYBM5M"


@pytest.mark.parametrize(
    "text, uri",
    [
        (f"spotify:track:{TRACK}", f"spotify:track:{TRACK}"),
        (f"  spotify:playlist:{PLAYLIST}\n", f"spotify:playlist:{PLAYLIST}"),
        (f"https://open.spotify.com/track/{TRACK}?si=a1b2c3", f"spotify:track:{TRACK}"),
        (f"https://open.spotify.com/intl-de/album/{TRACK}", f"spotify:album:{TRACK}"),
        (f"https://open.spotify.com/intl-pt-BR/artist/{TRACK}", f"spotify:artist:{TRACK}"),
        (f"https://open.spotify.com/episode/{TRACK}", f"spotify:episode:{TRACK}"),
        # What the AppleScript cannot play, or is no item's link at all.
        ("spotify:search:radiohead", None),
        (f"spotify:user:someone:playlist:{PLAYLIST}", None),
        (f"spotify:track:{TRACK[:-1]}", None),
        (f"spotify:track:{TRACK}x", None),
        (f"http://open.spotify.com/track/{TRACK}", None),
        (f"https://open.spotify.com.evil.example/track/{TRACK}", None),
        (f"https://open.spotify.com/track/{TRACK} and more", None),
        ("Paranoid Android", None),
    ],
)
def test_a_link_is_a_playable_spotify_uri_or_a_web_link_to_one_and_nothing_else(text: str, uri: str | None) -> None:
    assert link(text) == uri


class Osascript:
    """osascript, answering each script with the next of `answers`, and keeping what it was given."""

    def __init__(self, *answers: Ran) -> None:
        self.answers = list(answers)
        self.told: list[tuple[Sequence[str], Sequence[str]]] = []

    async def __call__(self, script: Sequence[str], argv: Sequence[str]) -> Ran:
        self.told.append((script, argv))
        return self.answers.pop(0)


def printed(text: str) -> Ran:
    return Ran(0, f"{text}\n".encode(), b"")


def tools(tell: Osascript, catalogue: Catalogue | None = None) -> dict[str, Body]:
    return {tool.name: tool.body for tool in spotify_tools(Player(tell), catalogue or Catalogue(Missing("none here")))}


async def test_now_playing_reads_the_track_spotify_is_on() -> None:
    tell = Osascript(printed("\x1f".join(["playing", "Paranoid Android", "Radiohead", "OK Computer", f"spotify:track:{TRACK}", "61", "386"])))
    assert await tools(tell)["spotify_now_playing"]() == {
        "state": "playing", "track": "Paranoid Android", "artist": "Radiohead", "album": "OK Computer", "link": f"spotify:track:{TRACK}", "position_seconds": 61, "duration_seconds": 386,
    }


async def test_now_playing_says_a_spotify_that_is_closed_or_stopped_has_no_track() -> None:
    assert await tools(Osascript(printed("not running")))["spotify_now_playing"]() == {"state": "not running"}
    assert await tools(Osascript(printed("stopped")))["spotify_now_playing"]() == {"state": "stopped"}


async def test_an_answer_hands_cannot_read_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(Refused, match="cannot read"):
        await Player(Osascript(printed("playing\x1fhalf"))).now_playing()


async def test_play_hands_spotify_the_uri_as_an_argument_never_inside_the_script() -> None:
    tell = Osascript(printed(""))
    assert await tools(tell)["spotify_play"](link=f"https://open.spotify.com/track/{TRACK}?si=x") == {"playing": f"spotify:track:{TRACK}"}
    [(script, argv)] = tell.told
    assert argv == [f"spotify:track:{TRACK}"] and not any(TRACK in line for line in script)


async def test_play_with_no_link_goes_on_with_what_was_playing() -> None:
    tell = Osascript(printed(""))
    assert await tools(tell)["spotify_play"]() == {"playing": "what was playing"}
    [(script, argv)] = tell.told
    assert 'tell application "Spotify" to play' in script and argv == []


async def test_play_refuses_what_is_no_spotify_link_and_tells_spotify_nothing() -> None:
    tell = Osascript()
    assert await tools(tell)["spotify_play"](link="Paranoid Android") == {"error": "'Paranoid Android' is no Spotify link; find what to play with spotify_search"}
    assert tell.told == []


async def test_a_command_to_a_spotify_that_is_not_running_is_refused_and_starts_nothing() -> None:
    body = tools(Osascript(printed("not running")))
    assert await body["spotify_pause"]() == {"error": "Spotify is not running"}


async def test_each_command_says_what_it_did() -> None:
    tell = Osascript(*(printed("done") for _ in range(5)))
    body = tools(tell)
    assert await body["spotify_pause"]() == {"paused": True}
    assert await body["spotify_skip"](to="previous") == {"skipped_to": "previous"}
    assert await body["spotify_volume"](level=30) == {"volume": 30}
    assert await body["spotify_shuffle"](on=True) == {"shuffle": True}
    assert await body["spotify_repeat"](on=False) == {"repeat": False}
    commands = [script[3] for script, _ in tell.told]
    assert commands == ["pause", "previous track", "set sound volume to (item 1 of argv) as integer", "set shuffling to true", "set repeating to false"]
    assert tell.told[2][1] == ("30",)


async def test_a_volume_off_spotifys_scale_is_refused_before_spotify_is_told() -> None:
    tell = Osascript()
    assert await tools(tell)["spotify_volume"](level=101) == {"error": "Spotify's volume runs from 0 to 100, not 101"}
    assert tell.told == []


async def test_macos_withholding_control_of_spotify_is_said_with_where_to_allow_it() -> None:
    denied = Ran(1, b"", b"execution error: Not authorized to send Apple events to Spotify. (-1743)\n")
    result = await tools(Osascript(denied))["spotify_pause"]()
    assert result == {"error": "macOS has not let hands control Spotify: allow it under System Settings, Privacy & Security, Automation"}


def test_the_catalogues_credentials_come_from_the_environment_and_their_absence_names_each_one_missing() -> None:
    assert credentials({"SPOTIFY_CLIENT_ID": " id ", "SPOTIFY_CLIENT_SECRET": "secret"}) == Credentials("id", "secret")
    missing = credentials({"SPOTIFY_CLIENT_ID": "id"})
    assert isinstance(missing, Missing) and "SPOTIFY_CLIENT_SECRET" in missing.why and "SPOTIFY_CLIENT_ID" not in missing.why
    both = credentials({})
    assert isinstance(both, Missing) and "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET" in both.why


async def test_a_search_without_credentials_is_refused_saying_how_to_get_them() -> None:
    result = await tools(Osascript(), Catalogue(credentials({})))["spotify_search"](query="radiohead")
    assert "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET" in str(result["error"])


@asynccontextmanager
async def spotify_api(granted: list[str], searched: list[dict[str, str]]) -> AsyncGenerator[str]:
    async def token(request: web.Request) -> web.Response:
        assert request.headers["Authorization"].startswith("Basic ") and (await request.post())["grant_type"] == "client_credentials"
        granted.append(f"token-{len(granted)}")
        return web.json_response({"access_token": granted[-1], "token_type": "Bearer", "expires_in": 3600})

    async def search(request: web.Request) -> web.Response:
        searched.append({**request.query, "token": request.headers["Authorization"]})
        match request.query["type"]:
            case "track":
                items: list[object] = [{"name": "Paranoid Android", "artists": [{"name": "Radiohead"}], "uri": f"spotify:track:{TRACK}"}]
            case _:
                items = [None, {"name": "Today's Top Hits", "owner": {"display_name": "Spotify"}, "uri": f"spotify:playlist:{PLAYLIST}"}]
        return web.json_response({f"{request.query['type']}s": {"items": items}})

    app = web.Application()
    app.router.add_post("/token", token)
    app.router.add_get("/search", search)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


async def test_a_search_holds_its_token_until_it_runs_out_and_its_event_says_which() -> None:
    granted: list[str] = []
    searched: list[dict[str, str]] = []
    now = [0.0]
    recorded: list[object] = []
    async with spotify_api(granted, searched) as url:
        catalogue = Catalogue(Credentials("id", "secret"), lambda: now[0], token_url=f"{url}/token", search_url=f"{url}/search")
        [search] = [audited(tool, recorded.append) for tool in spotify_tools(Player(Osascript()), catalogue) if tool.name == "spotify_search"]
        assert await search.body(query="paranoid android") == {"found": [{"name": "Paranoid Android", "by": "Radiohead", "link": f"spotify:track:{TRACK}"}]}
        # A playlist Spotify cannot show comes as a null, and is no find.
        assert await search.body(query="top hits", kind="playlist") == {"found": [{"name": "Today's Top Hits", "by": "Spotify", "link": f"spotify:playlist:{PLAYLIST}"}]}
        now[0] = 3600.0 - 59.0
        await search.body(query="ok computer")
    assert granted == ["token-0", "token-1"]
    assert [(query["q"], query["type"], query["limit"], query["token"]) for query in searched] == [
        ("paranoid android", "track", "5", "Bearer token-0"),
        ("top hits", "playlist", "5", "Bearer token-0"),
        ("ok computer", "track", "5", "Bearer token-1"),
    ]
    events = [event for event in recorded if isinstance(event, WideEvent)]
    assert [event.facts["spotify_token"] for event in events] == ["fetched", "held", "fetched"]


async def test_spotify_refusing_the_credentials_is_said_with_its_answer() -> None:
    async def token(_request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid_client"}, status=400)

    app = web.Application()
    app.router.add_post("/token", token)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    try:
        catalogue = Catalogue(Credentials("id", "wrong"), token_url=f"{url}/token", search_url=f"{url}/search")
        result = await tools(Osascript(), catalogue)["spotify_search"](query="radiohead")
    finally:
        await runner.cleanup()
    assert str(result["error"]).startswith("Spotify refused hands' developer app credentials (400)") and "invalid_client" in str(result["error"])
