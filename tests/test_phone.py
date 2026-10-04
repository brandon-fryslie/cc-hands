"""The phone's call, made by a page played here by aiortc over loopback: its microphone and button in, hands' speech out."""

import asyncio
import datetime
import json
import ssl
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from importlib import resources
from types import SimpleNamespace
from typing import cast, get_args

import numpy as np
import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
from loguru import logger
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.transports.local.audio import LocalAudioTransportParams

from hands.sessions.audit import Entry
from hands.voice.echo import EchoCanceller
from hands.voice.microphone import KeyedAudioTransport, Output, PortAudio
from hands.voice import phone as phone_module
from hands.voice.mark import Mark
from hands.voice.phone import Asked, CallDeclined, CallLeft, CallRefused, CallUnreached, Offer, Phone, kind
from hands.voice.transcript import Cut, Heard, Line, Saying, Spoken
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent, begun
from hands.voice import phonepage
from hands.voice.phoneaddress import RENEW_DAYS, Tailnet, Untailed, own_certificate, phone_key
from hands.voice.phonepage import phone_app
from hands.voice.ptt import KeyedAudio, PushToTalk
from conftest import events

KEY = "the-phone-key"
PAGE_RATE = 48000
# Twenty milliseconds of the page's microphone, as one message: loud, then the same length of silence.
LOUD = (np.full(PAGE_RATE // 50, 8000, dtype=np.int16)).tobytes()


@dataclass
class Page:
    """What the phone's page does, done by aiortc: a channel for its microphone and button, and an ear for hands."""

    peer: RTCPeerConnection
    channel: RTCDataChannel
    played: list[AudioFrame]
    told: list[object]
    opened: asyncio.Event

    def send(self, said: bytes | str) -> None:
        self.channel.send(said)


# The page every call is from unless a test names another.
FIRST = Asked("the first page", "take")


async def a_page(asked: Asked = FIRST) -> tuple[Page, Offer]:
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    peer.addTransceiver("audio", direction="recvonly")
    channel = peer.createDataChannel("talk", ordered=True)
    page = Page(peer, channel, [], [], asyncio.Event())
    channel.on("open", page.opened.set)
    channel.on("message", page.told.append)

    @peer.on("track")
    def heard(track: MediaStreamTrack) -> None:  # pyright: ignore[reportUnusedFunction]
        async def listen() -> None:
            while True:
                page.played.append(await track.recv())  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        asyncio.ensure_future(listen())

    await peer.setLocalDescription(await peer.createOffer())
    return page, Offer(peer.localDescription.sdp, "offer", PAGE_RATE, asked)


async def answered(phone: Phone, offer: Offer, remote: str) -> RTCSessionDescription:
    """hands' answer to an offer it takes."""
    match await phone.answer(offer, remote, begun()):
        case RTCSessionDescription() as answer:
            return answer
        case declined:
            pytest.fail(f"hands declined the offer: {declined}")


def calls(recorded: list[Entry]) -> list[WideEvent]:
    """Each call's one event, as each ended."""
    return [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "phone.call"]


def left(event: WideEvent) -> CallLeft:
    """How the call `event` is the event of ended, where it was up."""
    match event.facts["ended"]:
        case CallLeft() as ended:
            return ended
        case ended:
            pytest.fail(f"the call was not up as it ended: {ended}")


@dataclass
class Call:
    phone: Phone
    key: PushToTalk
    page: Page
    recorded: list[Entry]

    async def until(self, done: Callable[[], bool]) -> None:
        async with asyncio.timeout(5):
            while not done():
                await asyncio.sleep(0.01)

    def heard(self) -> list[KeyedAudio]:
        frames: list[KeyedAudio] = []
        while not self.phone.heard.empty():
            frames.append(self.phone.heard.get_nowait())
        return frames


@pytest.fixture
async def call() -> AsyncGenerator[Call, None]:
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    page, offer = await a_page()
    await page.peer.setRemoteDescription(await answered(phone, offer, "192.168.7.20"))
    async with asyncio.timeout(10):
        await page.opened.wait()
        while key.gate.place != "phone":
            await asyncio.sleep(0.01)
    yield Call(phone, key, page, recorded)
    await page.peer.close()
    await phone.stop()


async def test_a_call_puts_hands_at_the_phone(call: Call) -> None:
    assert call.key.gate.place == "phone"
    # A call's one event is written as it ends.
    assert call.recorded == []



async def test_the_moves_a_call_makes_are_part_of_its_trace() -> None:
    recorded: list[Entry] = []
    key = PushToTalk(recorded.append)
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    page, offer = await a_page()
    await page.peer.setRemoteDescription(await answered(phone, offer, "192.168.7.20"))
    async with asyncio.timeout(10):
        while key.gate.place != "phone":
            await asyncio.sleep(0.01)
    await phone.hang_up("hung up")
    [made] = calls(recorded)
    moves = [(move.facts["after"], move.trace_id, move.parent_id) for move in events(recorded, "place.moved")]
    assert moves == [("phone", made.trace_id, made.span_id), ("desk", made.trace_id, made.span_id)]
    await page.peer.close()
    await phone.stop()

async def test_a_turn_holds_what_was_said_while_the_button_was_down_and_nothing_else(call: Call) -> None:
    for said in [LOUD, "press", LOUD, LOUD, "release", LOUD]:
        call.page.send(said)
    await call.until(lambda: call.phone.moves.qsize() == 2 and call.phone.heard.qsize() >= 4)
    await asyncio.sleep(0.1)
    frames = call.heard()
    # Each message is resampled to the pipeline's rate, keyed by the button as it stood when the message arrived.
    keyed = [(frame.key, any(frame.audio)) for frame in frames]
    down = [loud for key, loud in keyed if key == "down"]
    assert down and all(down)
    assert all(not loud for key, loud in keyed if key == "up")
    first_down, last_down = [key for key, _ in keyed].index("down"), len(keyed) - 1 - [key for key, _ in keyed][::-1].index("down")
    assert all(key == "up" for key, _ in keyed[:first_down]) and all(key == "up" for key, _ in keyed[last_down + 1 :])
    assert all(frame.sample_rate == 16000 for frame in frames)
    assert [call.phone.moves.get_nowait(), call.phone.moves.get_nowait()] == ["start", "stop"]
    assert call.key.gate.key == "up"


async def test_the_desk_cannot_end_a_turn_held_at_the_phone(call: Call) -> None:
    call.page.send("press")
    await call.until(lambda: call.key.gate.key == "down")
    assert call.key.move("disarm", "held key") is None  # a Shift typed at the desk
    assert (call.key.gate.key, call.key.gate.place) == ("down", "phone")


async def test_what_hands_says_is_played_on_the_phone(call: Call) -> None:
    speech = (np.full(24000 // 10, 6000, dtype=np.int16)).tobytes()  # 100 ms at the played rate
    await asyncio.wait_for(call.phone.play(speech), 5)
    await call.until(lambda: any(np.abs(frame.to_ndarray()).max() > 1000 for frame in call.page.played))


# Every line of the transcript there is, as the page is told it.
LINES: list[Line] = [Heard("Is the parser fixed?"), Saying("It is.", 15.0), Spoken(), Saying("Its tests pass.", 12.5), Cut()]


async def test_the_page_is_told_each_mark_of_its_turn_and_each_line_of_the_transcript_and_the_call_counts_them(call: Call) -> None:
    for mark in get_args(Mark):
        call.phone.tell(mark)
    for line in LINES:
        call.phone.tell(line)
    await call.until(lambda: len(call.page.told) == len(get_args(Mark)) + len(LINES))
    assert [json.loads(cast(str, told)) for told in call.page.told] == [{"kind": "mark", "mark": mark} for mark in get_args(Mark)] + [
        {"kind": "heard", "text": "Is the parser fixed?"},
        {"kind": "saying", "text": "It is.", "chars_per_sec": 15.0},
        {"kind": "spoken"},
        {"kind": "saying", "text": "Its tests pass.", "chars_per_sec": 12.5},
        {"kind": "cut"},
    ]
    await call.phone.hang_up("stopped")
    [event] = calls(call.recorded)
    assert (left(event).told, left(event).lines) == (len(get_args(Mark)), len(LINES))


async def test_a_mark_with_no_call_up_is_told_to_nobody() -> None:
    """A turn taken at the desk passes the same marks, and no page is waiting on it."""
    phone = Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=24000, record=lambda _: None)
    phone.tell("released")


def test_the_page_knows_every_mark_hands_tells_it() -> None:
    """The page says what each mark means by name, and names one it does not know as an error on its face."""
    page = resources.files("hands.voice").joinpath("phone.html").read_text()
    table = page[page.index("const TOLD = {") : page.index("};", page.index("const TOLD = {"))]
    assert [mark for mark in get_args(Mark) if f'"{mark}": ' not in table] == []


def test_the_page_knows_every_line_of_the_transcript_hands_tells_it() -> None:
    page = resources.files("hands.voice").joinpath("phone.html").read_text()
    table = page[page.index("const LINES = {") : page.index("\n};", page.index("const LINES = {"))]
    assert [kind(line) for line in get_args(Line) if f'"{kind(line)}": ' not in table] == []


async def test_the_page_hanging_up_puts_hands_back_at_the_desk_and_ends_what_was_playing(call: Call) -> None:
    call.page.send("press")
    await call.until(lambda: call.key.gate.key == "down")
    playing = call.phone.play(bytes(24000 * 2 * 60))  # a minute that will never be heard
    call.page.channel.close()
    await call.until(lambda: call.key.gate.place == "desk")
    # The hold open at the phone is thrown away, not sent.
    assert (call.key.gate.key, call.key.gate.dropped) == ("up", 1)
    await asyncio.wait_for(playing, 1)
    [event] = calls(call.recorded)
    assert (event.outcome, event.parent_id, event.facts["remote"]) == ("ok", None, "192.168.7.20")
    ended = left(event)
    assert ended.reason == "hung up" and 0 < ended.arrived_ms <= event.duration_ms
    # A call whose page was told nothing says so: zero, written down.
    assert (ended.told, ended.lines) == (0, 0)


async def test_a_newer_call_replaces_the_one_before_it_once_it_connects(call: Call) -> None:
    page, offer = await a_page()
    answer = await answered(call.phone, offer, "100.66.66.10")
    # Answered, not yet connected: the call that is up is still the one hands is at.
    assert call.recorded == []
    await page.peer.setRemoteDescription(answer)
    await call.until(lambda: bool(call.recorded))
    [replaced] = calls(call.recorded)
    assert (replaced.outcome, replaced.facts["remote"], left(replaced).reason) == ("ok", "192.168.7.20", "replaced")
    assert call.key.gate.place == "phone"
    await page.peer.close()


async def test_an_offer_that_never_connects_moves_nothing_and_is_let_go_of_by_the_next() -> None:
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    unheard, offer = await a_page()
    await phone.answer(offer, "192.168.7.21", begun())  # its page never takes the answer
    assert key.gate.place == "desk" and recorded == []
    page, offer = await a_page()
    await phone.answer(offer, "192.168.7.22", begun())
    [older] = calls(recorded)
    assert (older.outcome, dict(older.facts)) == ("ok", {"remote": "192.168.7.21", "asked": FIRST, "ended": CallUnreached("replaced")})
    await phone.stop()
    [_, newer] = calls(recorded)
    assert (newer.outcome, dict(newer.facts)) == ("ok", {"remote": "192.168.7.22", "asked": FIRST, "ended": CallUnreached("stopped")})
    await unheard.peer.close()
    await page.peer.close()


async def connected(phone: Phone, asked: Asked) -> Page:
    """A page `asked` called, answered, and hands at it."""
    page, offer = await a_page(asked)
    await page.peer.setRemoteDescription(await answered(phone, offer, "192.168.7.30"))
    async with asyncio.timeout(10):
        while not (page.opened.is_set() and phone._call is not None and phone._call.asked == asked):  # pyright: ignore[reportPrivateUsage]
            await asyncio.sleep(0.01)
    return page


async def test_a_page_resuming_its_own_dropped_call_is_answered_and_takes_its_place(call: Call) -> None:
    """hands may not yet have noticed the call the page lost: the page's own stale call gives way to it."""
    page = await connected(call.phone, Asked(FIRST.page, "resume"))
    [stale] = calls(call.recorded)
    assert (stale.facts["asked"], left(stale).reason) == (FIRST, "replaced")
    assert call.key.gate.place == "phone"
    await page.peer.close()


async def test_a_page_resuming_while_another_page_has_the_call_is_declined_and_moves_nothing(call: Call) -> None:
    other = Asked("another page", "resume")
    _, offer = await a_page(other)
    assert await call.phone.answer(offer, "192.168.7.31", begun()) == CallDeclined()
    [declined] = calls(call.recorded)
    assert (declined.outcome, dict(declined.facts)) == ("ok", {"remote": "192.168.7.31", "asked": other, "ended": CallDeclined()})
    # The page that had the call still has it, and still speaks for it.
    assert call.key.gate.place == "phone"
    call.page.send("press")
    await call.until(lambda: call.key.gate.key == "down")


async def test_a_page_resuming_while_another_page_has_an_offer_in_is_declined() -> None:
    """The user just pressed Connect on the other page: a page calling again on its own never takes that from them."""
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    pressed, offer = await a_page()
    await phone.answer(offer, "192.168.7.32", begun())
    _, resumed = await a_page(Asked("another page", "resume"))
    assert await phone.answer(resumed, "192.168.7.33", begun()) == CallDeclined()
    assert phone._offered is not None and phone._offered.asked == FIRST  # pyright: ignore[reportPrivateUsage]
    await phone.stop()
    await pressed.peer.close()


async def test_a_resume_answered_alongside_a_press_never_takes_the_phone_from_it() -> None:
    """A press put in while a resume is still being answered is the page in the user's hand, whichever finishes first."""
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    resumed, resume = await a_page(Asked("another page", "resume"))
    pressed, take = await a_page()
    # The resume finishes answering after the press is put in.
    _, declined = await asyncio.gather(phone.answer(take, "192.168.7.35", begun()), phone.answer(resume, "192.168.7.34", begun()))
    assert declined == CallDeclined()
    assert phone._offered is not None and phone._offered.asked == FIRST  # pyright: ignore[reportPrivateUsage]
    await phone.stop()
    await resumed.peer.close()
    await pressed.peer.close()

async def test_a_page_resuming_with_no_call_up_is_answered() -> None:
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    asked = Asked("a page hands restarted under", "resume")
    page = await connected(phone, asked)
    await phone.hang_up("stopped")
    [event] = calls(recorded)
    assert (event.facts["asked"], left(event).reason) == (asked, "stopped")
    await page.peer.close()


async def test_a_page_taking_the_phone_takes_it_from_another_page(call: Call) -> None:
    page = await connected(call.phone, Asked("another page", "take"))
    [replaced] = calls(call.recorded)
    assert (replaced.facts["asked"], left(replaced).reason) == (FIRST, "replaced")
    await page.peer.close()


async def test_a_message_that_is_neither_audio_nor_the_button_is_said_and_moves_nothing(call: Call) -> None:
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        call.page.send("wave")
        await call.until(lambda: bool(errors))
    finally:
        logger.remove(sink)
    assert "'wave'" in errors[0]
    assert call.key.gate.key == "up"


Served = tuple[TestClient[web.Request, web.Application], Phone, PushToTalk, list[Entry]]


def offered(offer: Offer) -> str:
    """`offer` as the page posts it."""
    return json.dumps({"sdp": offer.sdp, "type": "offer", "rate": offer.rate, "page": offer.asked.page, "claim": offer.asked.claim})


@pytest.fixture
async def page_server() -> AsyncGenerator[Served, None]:
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    client = TestClient(TestServer(phone_app(phone, KEY, recorded.append)))
    await client.start_server()
    yield client, phone, key, recorded
    await phone.stop()
    await client.close()



async def test_the_page_is_served_to_anyone(page_server: Served) -> None:
    client, _, _, _ = page_server
    shown = await client.get("/")
    assert shown.status == 200 and "Hold to talk" in await shown.text()


async def test_an_offer_without_the_phone_key_is_refused_and_recorded(page_server: Served) -> None:
    client, phone, key, recorded = page_server
    page, offer = await a_page()
    body = offered(offer)
    refused = await client.post("/offer", data=body, headers={"Authorization": "Bearer guessed"})
    assert refused.status == 401
    [event] = calls(recorded)
    assert (event.outcome, event.error, event.parent_id) == ("failed", "offered without the phone's key", None)
    assert dict(event.facts) == {"remote": "127.0.0.1", "ended": CallRefused("offered without the phone's key")}
    assert phone.moves.empty() and phone.heard.empty()
    taken = await client.post("/offer", data=body, headers={"Authorization": f"Bearer {KEY}"})
    assert taken.status == 200
    await page.peer.setRemoteDescription(RTCSessionDescription(**await taken.json()))
    async with asyncio.timeout(10):
        while key.gate.place != "phone":
            await asyncio.sleep(0.01)
    await phone.hang_up("hung up")
    [_, up] = calls(recorded)
    assert (up.facts["remote"], left(up).reason) == ("127.0.0.1", "hung up")
    await page.peer.close()


async def test_an_offer_that_is_no_offer_is_refused_and_is_one_event_saying_why(page_server: Served) -> None:
    client, _, _, recorded = page_server
    refused = await client.post("/offer", data="{}", headers={"Authorization": f"Bearer {KEY}"})
    assert refused.status == 400
    [event] = calls(recorded)
    assert (event.outcome, event.error, event.facts["ended"]) == ("failed", await refused.text(), CallRefused(await refused.text()))


@pytest.mark.parametrize(
    "body",
    [
        {"sdp": "v=0", "type": "offer", "rate": PAGE_RATE},
        {"sdp": "v=0", "type": "offer", "rate": PAGE_RATE, "page": "", "claim": "take"},
        {"sdp": "v=0", "type": "offer", "rate": PAGE_RATE, "page": "a page", "claim": "steal"},
    ],
    ids=["no page or claim", "a page with no name", "a claim that is neither"],
)
async def test_an_offer_that_does_not_say_which_page_asks_what_is_refused(page_server: Served, body: dict[str, object]) -> None:
    client, _, _, recorded = page_server
    refused = await client.post("/offer", data=json.dumps(body), headers={"Authorization": f"Bearer {KEY}"})
    assert refused.status == 400
    [event] = calls(recorded)
    assert event.facts["ended"] == CallRefused(await refused.text())


async def test_a_resume_the_page_server_declines_is_a_conflict_the_page_can_tell_apart(page_server: Served) -> None:
    client, _, key, recorded = page_server
    taker, offer = await a_page()
    taken = await client.post("/offer", data=offered(offer), headers={"Authorization": f"Bearer {KEY}"})
    await taker.peer.setRemoteDescription(RTCSessionDescription(**await taken.json()))
    async with asyncio.timeout(10):
        while key.gate.place != "phone":
            await asyncio.sleep(0.01)
    _, resumed = await a_page(Asked("another page", "resume"))
    declined = await client.post("/offer", data=offered(resumed), headers={"Authorization": f"Bearer {KEY}"})
    assert declined.status == 409
    assert calls(recorded)[-1].facts["ended"] == CallDeclined()
    await taker.peer.close()


async def test_an_offer_whose_body_cannot_be_read_is_one_failed_event_saying_what_raised(page_server: Served) -> None:
    client, _, _, recorded = page_server
    unread = await client.post("/offer", data=b"\xff\xfe", headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json; charset=utf-8"})
    assert unread.status == 500
    [event] = calls(recorded)
    assert (event.outcome, dict(event.facts)) == ("failed", {"remote": "127.0.0.1"})
    assert event.error is not None and event.error.startswith("UnicodeDecodeError") and event.trace


async def test_an_offer_hands_cannot_answer_is_one_failed_event_and_raises() -> None:
    recorded: list[Entry] = []
    phone = Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=24000, record=recorded.append)
    with pytest.raises(ValueError):
        await phone.answer(Offer("not a session description", "offer", PAGE_RATE, FIRST), "192.168.7.23", begun())
    [event] = calls(recorded)
    assert (event.outcome, dict(event.facts)) == ("failed", {"remote": "192.168.7.23", "asked": FIRST})
    assert event.error is not None and event.error.startswith("ValueError") and event.trace


class Desk:
    """The desk's speaker stream, kept: what was written to it."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, audio: bytes) -> None:
        self.written.append(audio)

    def get_output_latency(self) -> float:
        return 0.0

    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


def a_transport(call: Call) -> tuple[KeyedAudioTransport, Desk, list[bytes]]:
    """The keyed transport on the call's gate and phone, its desk stream kept and its pushed audio kept."""
    echo = EchoCanceller()
    transport = KeyedAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True), call.key, call.phone, lambda _: None, echo=lambda: echo)
    desk, pushed = Desk(), list[bytes]()
    transport.output().attach(cast(PortAudio, SimpleNamespace()), Output(desk, "MacBook Pro Speakers", echo))
    microphone = transport.input()
    microphone._sample_rate = 16000  # pyright: ignore[reportPrivateUsage]  # as setup sets it

    async def push_audio_frame(frame: InputAudioRawFrame) -> None:
        pushed.append(frame.audio)

    microphone.push_audio_frame = push_audio_frame
    microphone.get_event_loop = asyncio.get_running_loop
    return transport, desk, pushed


async def test_while_hands_is_at_the_phone_the_desk_is_neither_heard_nor_played_to(call: Call) -> None:
    transport, desk, pushed = a_transport(call)
    echo = cast(Output, transport.output().opened).echo  # the canceller the desk streams were attached with
    call.key.move("start", "held key")  # a turn opened at the desk takes hands there...
    assert call.key.gate.place == "desk"
    call.page.send("press")  # ...and one opened at the phone takes it back
    await call.until(lambda: call.key.gate.place == "phone")
    transport.input()._captured(echo, b"\x7f\x7f" * 320, 320, None, 0)  # pyright: ignore[reportPrivateUsage]
    await asyncio.sleep(0.05)
    assert pushed == []
    speech = (np.full(24000 // 25, 6000, dtype=np.int16)).tobytes()
    assert await asyncio.wait_for(transport.output().write_audio_frame(OutputAudioRawFrame(audio=speech, sample_rate=24000, num_channels=1)), 5)
    assert desk.written == []
    await call.until(lambda: any(np.abs(frame.to_ndarray()).max() > 1000 for frame in call.page.played))


async def test_a_page_that_goes_quiet_is_hung_up_and_hands_is_at_the_desk_again(call: Call, monkeypatch: pytest.MonkeyPatch) -> None:
    # A browser closed outright says nothing; its microphone simply stops arriving.
    monkeypatch.setattr(phone_module, "QUIET_SECS", 0.2)
    call.page.send(LOUD)
    await call.until(lambda: call.key.gate.place == "desk")
    [event] = calls(call.recorded)
    assert left(event).reason == "went quiet"


async def _served(home: Home, net: Tailnet | Untailed, monkeypatch: pytest.MonkeyPatch) -> list[Entry]:
    """What serving the page records, on a loopback port the system picks, with Tailscale answering `net`."""

    async def asked(_home: Home) -> Tailnet | Untailed:
        return net

    monkeypatch.setattr(phonepage, "PHONE_HOST", "127.0.0.1")
    monkeypatch.setattr(phonepage, "PHONE_PORT", 0)
    monkeypatch.setattr(phonepage, "tailnet", asked)
    recorded: list[Entry] = []
    phone = Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=24000, record=recorded.append)
    serving = asyncio.create_task(phonepage.serve_phone(phone, home, recorded.append))
    async with asyncio.timeout(5):
        while not recorded:
            await asyncio.sleep(0.01)
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    await phone.stop()
    return recorded


async def test_the_page_served_under_the_tailnet_name_is_one_event_naming_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    cert, key = own_certificate(home, datetime.datetime.now(datetime.UTC))
    [event] = await _served(home, Tailnet("hands.example.ts.net", cert, key), monkeypatch)
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("phone.served", "ok")
    # The port the system bound, not the 0 it was asked for.
    port = event.facts["port"]
    assert dict(event.facts) == {"port": port, "tailnet": "hands.example.ts.net"} and isinstance(port, int) and port > 0


async def test_the_page_served_on_the_lan_alone_is_one_event_saying_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    [event] = await _served(Home(tmp_path), Untailed("the tailscale command is not on the PATH"), monkeypatch)
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("phone.served", "ok")
    port = event.facts["port"]
    assert dict(event.facts) == {"port": port, "untailed": "the tailscale command is not on the PATH"} and isinstance(port, int) and port > 0


async def test_a_call_to_the_served_page_is_the_root_of_a_trace_of_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The page's requests run in a context copied inside `phone.served`, which ended as the page came up.
    async def asked(_home: Home) -> Untailed:
        return Untailed("the tailscale command is not on the PATH")

    monkeypatch.setattr(phonepage, "PHONE_HOST", "127.0.0.1")
    monkeypatch.setattr(phonepage, "PHONE_PORT", 0)
    monkeypatch.setattr(phonepage, "tailnet", asked)
    recorded: list[Entry] = []
    phone = Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=24000, record=recorded.append)
    serving = asyncio.create_task(phonepage.serve_phone(phone, Home(tmp_path), recorded.append))
    async with asyncio.timeout(5):
        while not recorded:
            await asyncio.sleep(0.01)
    [served] = recorded
    assert isinstance(served, WideEvent)
    unverified = ssl.create_default_context()
    unverified.check_hostname, unverified.verify_mode = False, ssl.CERT_NONE
    async with ClientSession() as session, session.post(f"https://127.0.0.1:{served.facts['port']}/offer", data="{}", ssl=unverified) as refused:
        assert refused.status == 401
    [event] = calls(recorded)
    assert event.parent_id is None and event.trace_id != served.trace_id
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await serving
    await phone.stop()


def test_the_phone_key_is_made_once_readable_by_the_user_alone(tmp_path: Path) -> None:
    home = Home(tmp_path)
    key = phone_key(home)
    assert key and phone_key(home) == key
    assert (home.phone / "key").stat().st_mode & 0o777 == 0o600
    # Nothing is left beside it from its making.
    assert [path.name for path in home.phone.iterdir()] == ["key"]


def test_a_key_file_that_holds_no_key_lets_no_call_in(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.phone.mkdir(parents=True)
    (home.phone / "key").write_text("\n")
    with pytest.raises(Rejected, match="holds no key"):
        phone_key(home)


def test_hands_own_certificate_is_kept_until_near_its_end_and_made_again_then(tmp_path: Path) -> None:
    home = Home(tmp_path)
    made = datetime.datetime(2026, 10, 3, tzinfo=datetime.UTC)
    cert, key = own_certificate(home, made)
    first = cert.read_bytes()
    assert own_certificate(home, made + datetime.timedelta(days=300)) == (cert, key) and cert.read_bytes() == first
    own_certificate(home, made + datetime.timedelta(days=397 - RENEW_DAYS + 1))
    assert cert.read_bytes() != first
    # The certificate and its key are a pair a server loads.
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)
    assert key.stat().st_mode & 0o777 == 0o600
