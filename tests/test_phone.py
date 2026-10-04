"""The phone's call, made by a page played here by aiortc over loopback: its microphone and button in, hands' speech out."""

import asyncio
import datetime
import json
import socket
import ssl
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from loguru import logger
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.transports.local.audio import LocalAudioTransportParams

from hands.sessions.audit import Entry, PhoneArrived, PhoneLeft, PhoneRefused, PhoneUnreached
from hands.voice.echo import EchoCanceller
from hands.voice.microphone import KeyedAudioTransport, Output, PortAudio
from hands.voice import phone as phone_module
from hands.voice.phone import Offer, Phone
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wide import WideEvent
from hands.voice import phonepage
from hands.voice.phonepage import RENEW_DAYS, Tailnet, Untailed, own_certificate, phone_app, phone_key
from hands.voice.ptt import KeyedAudio, PushToTalk

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
    opened: asyncio.Event

    def send(self, said: bytes | str) -> None:
        self.channel.send(said)


async def a_page() -> tuple[Page, Offer]:
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    peer.addTransceiver("audio", direction="recvonly")
    channel = peer.createDataChannel("talk", ordered=True)
    page = Page(peer, channel, [], asyncio.Event())
    channel.on("open", page.opened.set)

    @peer.on("track")
    def heard(track: MediaStreamTrack) -> None:  # pyright: ignore[reportUnusedFunction]
        async def listen() -> None:
            while True:
                page.played.append(await track.recv())  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        asyncio.ensure_future(listen())

    await peer.setLocalDescription(await peer.createOffer())
    return page, Offer(peer.localDescription.sdp, "offer", PAGE_RATE)


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
    answer = await phone.answer(offer, "192.168.7.20")
    await page.peer.setRemoteDescription(answer)
    async with asyncio.timeout(10):
        await page.opened.wait()
        while key.gate.place != "phone":
            await asyncio.sleep(0.01)
    yield Call(phone, key, page, recorded)
    await page.peer.close()
    await phone.stop()


async def test_a_call_puts_hands_at_the_phone(call: Call) -> None:
    assert call.key.gate.place == "phone"
    assert call.recorded == [PhoneArrived(remote="192.168.7.20")]


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


async def test_the_page_hanging_up_puts_hands_back_at_the_desk_and_ends_what_was_playing(call: Call) -> None:
    call.page.send("press")
    await call.until(lambda: call.key.gate.key == "down")
    playing = call.phone.play(bytes(24000 * 2 * 60))  # a minute that will never be heard
    call.page.channel.close()
    await call.until(lambda: call.key.gate.place == "desk")
    # The hold open at the phone is thrown away, not sent.
    assert (call.key.gate.key, call.key.gate.dropped) == ("up", 1)
    await asyncio.wait_for(playing, 1)
    assert isinstance(call.recorded[-1], PhoneLeft) and call.recorded[-1].reason == "hung up"


async def test_a_newer_call_replaces_the_one_before_it_once_it_connects(call: Call) -> None:
    page, offer = await a_page()
    answer = await call.phone.answer(offer, "100.66.66.10")
    # Answered, not yet connected: the call that is up is still the one hands is at.
    assert [type(entry).__name__ for entry in call.recorded] == ["PhoneArrived"]
    await page.peer.setRemoteDescription(answer)
    await call.until(lambda: len(call.recorded) == 3)
    assert [type(entry).__name__ for entry in call.recorded] == ["PhoneArrived", "PhoneLeft", "PhoneArrived"]
    assert call.recorded[-1] == PhoneArrived(remote="100.66.66.10")
    assert call.key.gate.place == "phone"
    await page.peer.close()


async def test_an_offer_that_never_connects_moves_nothing_and_is_let_go_of_by_the_next() -> None:
    key, recorded = PushToTalk(lambda _: None), list[Entry]()
    phone = Phone(key, heard_rate=16000, played_rate=24000, record=recorded.append)
    unheard, offer = await a_page()
    await phone.answer(offer, "192.168.7.21")  # its page never takes the answer
    assert key.gate.place == "desk" and recorded == []
    page, offer = await a_page()
    await phone.answer(offer, "192.168.7.22")
    match recorded:
        case [PhoneUnreached(remote="192.168.7.21", reason="replaced")]:
            pass
        case _:
            pytest.fail(f"the older offer was not let go of as replaced: {recorded}")
    await phone.stop()
    assert isinstance(recorded[-1], PhoneUnreached) and recorded[-1].reason == "stopped"
    await unheard.peer.close()
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


Served = tuple[TestClient[web.Request, web.Application], Phone, list[Entry]]


@pytest.fixture
async def page_server() -> AsyncGenerator[Served, None]:
    recorded: list[Entry] = []
    phone = Phone(PushToTalk(lambda _: None), heard_rate=16000, played_rate=24000, record=recorded.append)
    client = TestClient(TestServer(phone_app(phone, KEY, recorded.append)))
    await client.start_server()
    yield client, phone, recorded
    await phone.stop()
    await client.close()



async def test_the_page_is_served_to_anyone(page_server: Served) -> None:
    client, _, _ = page_server
    shown = await client.get("/")
    assert shown.status == 200 and "Hold to talk" in await shown.text()


async def test_an_offer_without_the_phone_key_is_refused_and_recorded(page_server: Served) -> None:
    client, phone, recorded = page_server
    page, offer = await a_page()
    body = json.dumps({"sdp": offer.sdp, "type": "offer", "rate": PAGE_RATE})
    refused = await client.post("/offer", data=body, headers={"Authorization": "Bearer guessed"})
    assert refused.status == 401
    assert isinstance(recorded[-1], PhoneRefused)
    assert phone.moves.empty() and not any(isinstance(entry, PhoneArrived) for entry in recorded)
    taken = await client.post("/offer", data=body, headers={"Authorization": f"Bearer {KEY}"})
    assert taken.status == 200
    await page.peer.setRemoteDescription(RTCSessionDescription(**await taken.json()))
    async with asyncio.timeout(10):
        while not isinstance(recorded[-1], PhoneArrived):
            await asyncio.sleep(0.01)
    await page.peer.close()


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
    left = call.recorded[-1]
    assert isinstance(left, PhoneLeft) and left.reason == "went quiet"


async def _served(home: Home, net: Tailnet | Untailed, monkeypatch: pytest.MonkeyPatch) -> tuple[int, list[Entry]]:
    """What serving the page records, on a free loopback port, with Tailscale answering `net`."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async def asked(_home: Home) -> Tailnet | Untailed:
        return net

    monkeypatch.setattr(phonepage, "PHONE_HOST", "127.0.0.1")
    monkeypatch.setattr(phonepage, "PHONE_PORT", port)
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
    return port, recorded


async def test_the_page_served_under_the_tailnet_name_is_one_event_naming_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    cert, key = own_certificate(home, datetime.datetime.now(datetime.UTC))
    port, recorded = await _served(home, Tailnet("hands.example.ts.net", cert, key), monkeypatch)
    [event] = recorded
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("phone.served", "ok")
    assert dict(event.facts) == {"port": port, "tailnet": "hands.example.ts.net"}


async def test_the_page_served_on_the_lan_alone_is_one_event_saying_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port, recorded = await _served(Home(tmp_path), Untailed("the tailscale command is not on the PATH"), monkeypatch)
    [event] = recorded
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("phone.served", "ok")
    assert dict(event.facts) == {"port": port, "untailed": "the tailscale command is not on the PATH"}


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
