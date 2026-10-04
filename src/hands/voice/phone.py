"""The phone: one WebRTC call from the page hands serves, carrying the phone's microphone in, hands' speech out, and
the page's talk button.

The call is where the phone is. While one is up hands is at the phone: a call arriving moves the gate there, and the
call ending moves it back to the desk, in the same step that ends the call, so nothing is ever played to a phone that
has gone. A newer call replaces the one before it: the page last opened is the one in the user's hand.

[LAW:nothing-unseen] a call is one unit of work, `phone.call`, from its page's offer to its end: refused at the page,
let go before it connected, or left after hands was at it, and why; its duration is how long it lasted.

The page sends its microphone as plain 16-bit audio over the call's data channel, in order with its button, and hands
plays to it over an audio track. Earbuds keep hands' voice out of the phone's microphone, and the page asks the browser
for its echo cancellation as well; so the phone's audio is gated by its button alone, with none of the desk
microphone's wait for the room to go quiet.
"""

import asyncio
import fractions
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler
from loguru import logger

from hands.sessions.audit import Record
from hands.sessions.payload import Rejected
from hands.sessions.wide import Begun, continuing, ended, since
from hands.voice.hold import Move
from hands.voice.ptt import KeyedAudio, PushToTalk

# The length of each frame sent to the phone: Opus's own 20 ms.
FRAME_SECS = 0.02
# How long the page may send nothing before its call is taken to be over. The page sends its microphone every 20 ms
# for as long as it is open, so a channel this quiet is a page closed, a phone put to sleep, or a network gone: measured
# 2026-10-03, a browser closed outright leaves aiortc's connection "connected" for good.
QUIET_SECS = 3.0


# Why a call ended: a newer call took its place, the page hung up, the connection failed, the page stopped sending
# (closed outright, or the phone asleep), or hands stopped.
PhoneGone = Literal["replaced", "hung up", "failed", "went quiet", "stopped"]


@dataclass(frozen=True)
class CallRefused:
    """The page's offer was not taken: it came without the phone's key, or was no offer."""

    why: str


@dataclass(frozen=True)
class CallUnreached:
    """The call was answered and let go before it connected: hands never moved to it."""

    reason: PhoneGone


@dataclass(frozen=True)
class CallLeft:
    """The call was up, hands at the phone from `arrived_ms` after its offer, and it ended."""

    reason: PhoneGone
    arrived_ms: float


# How a call ended; or what raised as hands answered it.
CallEnd = CallRefused | CallUnreached | CallLeft | BaseException


def call_ended(record: Record, began: Begun, remote: str, end: CallEnd) -> None:
    """[LAW:single-enforcer] the one place a call's event is written, as it ends, however it ends: failed where it was
    refused, where its connection failed, or where answering it raised."""
    match end:
        case BaseException():
            ended("phone.call", record, began, end, remote=remote)
        case CallRefused(why=why):
            ended("phone.call", record, began, None, why, remote=remote, ended=end)
        case CallUnreached(reason=reason) | CallLeft(reason=reason):
            failure = "the call's connection failed" if reason == "failed" else None
            ended("phone.call", record, began, None, failure, remote=remote, ended=end)


class Outbound(MediaStreamTrack):
    """Hands' speech as the phone hears it: what was given, in order, in real time, and silence between.

    Each piece given is a future resolved once it has been sent, so the speaker that gives it waits as long as the
    audio lasts, as it waits on a device's write; a call that ends resolves every future it still holds.
    """

    kind = "audio"

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._samples = round(sample_rate * FRAME_SECS)
        self._pending: deque[tuple[bytes, asyncio.Future[None] | None]] = deque()
        self._sent = 0
        self._start: float | None = None

    def give(self, audio: bytes) -> asyncio.Future[None]:
        """Queue 16-bit mono audio to be sent; the future resolves when its last frame has gone."""
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        size = self._samples * 2
        # The last frame is padded with silence: the phone hears a frame's worth at a time.
        padded = audio + bytes(-len(audio) % size)
        frames = [padded[i : i + size] for i in range(0, len(padded), size)]
        match self.readyState:
            case "ended":
                done.set_result(None)
            case _:
                for index, frame in enumerate(frames):
                    self._pending.append((frame, done if index == len(frames) - 1 else None))
                if not frames:
                    done.set_result(None)
        return done

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        # [LAW:no-ambient-temporal-coupling] frames are dated by the count sent, against the clock at the first: a
        # late frame is followed at once by the next, so the call never drifts behind real time.
        now = time.monotonic()
        start = self._start = now if self._start is None else self._start
        await asyncio.sleep(max(0.0, start + self._sent * FRAME_SECS - now))
        frame, done = self._pending.popleft() if self._pending else (bytes(self._samples * 2), None)
        if done is not None and not done.done():
            done.set_result(None)
        out = AudioFrame.from_ndarray(np.frombuffer(frame, dtype=np.int16)[None, :], format="s16", layout="mono")
        out.sample_rate = self._sample_rate
        out.pts = self._sent * self._samples
        out.time_base = fractions.Fraction(1, self._sample_rate)
        self._sent += 1
        return out

    def stop(self) -> None:
        super().stop()
        while self._pending:
            _, done = self._pending.popleft()
            if done is not None and not done.done():
                done.set_result(None)


@dataclass(frozen=True)
class Offer:
    """A page's session description, as its browser made it, every candidate gathered; and the rate its microphone's
    audio comes at, which is the browser's own."""

    sdp: str
    type: Literal["offer"]
    rate: int


# What the page sends over the call's data channel: its microphone, and its button.
Said = bytes | Move


def parse_said(message: object) -> Said:
    """One message of the page's: 16-bit mono audio at the offer's rate, or the button pressed or released; raises
    Rejected for anything else."""
    match message:
        case bytes():
            return message
        case "press":
            return "start"
        case "release":
            return "stop"
        case _:
            raise Rejected(f"the phone's page sent {message!r}, which is neither audio nor a press or a release")


@dataclass
class _Call:
    peer: RTCPeerConnection
    outbound: Outbound
    resampler: AudioResampler
    rate: int
    remote: str
    began: Begun
    # Set by every message the page sends; the watch hangs up a call that goes QUIET_SECS without one.
    heard: asyncio.Event
    # Set as it arrives: its watch, and how long after its offer it came.
    watch: asyncio.Task[None] | None = None
    arrived_ms: float = 0.0


class Phone:
    """The one call to the phone, and what it hears, plays, and presses.

    A call is offered once its page's offer is answered, and is where hands is once its channel opens, the page's
    first word: hands never moves to a call that cannot yet hear it, and an offer that never connects leaves the call
    that is up as it was. The newest offer is the page in the user's hand, so it lets go of any older one not yet up.

    [LAW:no-ambient-temporal-coupling] the microphone and the button come over one ordered channel, and each message is
    acted on as it arrives: a press moves the gate before the audio sent after it is keyed, and a release after the
    audio sent before it, so a turn holds exactly what was said while the button was down, whatever the network did.
    Every handler acts only for the call it was made for while that call is still the one it was: a word from a call
    already gone moves nothing.
    """

    def __init__(self, key: PushToTalk, heard_rate: int, played_rate: int, record: Record) -> None:
        self._key = key
        self._heard_rate = heard_rate
        self._played_rate = played_rate
        self._record = record
        # The call hands is at, and the newest offer answered and not yet up.
        self._call: _Call | None = None
        self._offered: _Call | None = None
        # The phone's microphone as the pipeline hears it: keyed frames at the pipeline's rate, while hands is at the phone.
        self.heard: asyncio.Queue[KeyedAudio] = asyncio.Queue()
        # The moves the page's button has made, as the gate took them, for what follows a move: its tone and its words.
        self.moves: asyncio.Queue[Move] = asyncio.Queue()

    async def answer(self, offer: Offer, remote: str, began: Begun) -> RTCSessionDescription:
        """Answer the call a page offers, `began` as it was offered, with every candidate hands has; it is taken once it
        connects."""
        # [LAW:one-source-of-truth] no ICE server: the phone reaches hands at an address it already has, on the LAN or
        # the tailnet, so hands' own addresses are every candidate there is.
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        outbound = Outbound(self._played_rate)
        resampler = AudioResampler(format="s16", layout="mono", rate=self._heard_rate)
        call = _Call(peer, outbound, resampler, offer.rate, remote, began, asyncio.Event())
        peer.addTrack(outbound)

        @peer.on("datachannel")
        async def opened(channel: RTCDataChannel) -> None:  # pyright: ignore[reportUnusedFunction]
            # The page closes its channel as it goes: the plainest word that the user hung up.
            @channel.on("close")
            async def closed() -> None:  # pyright: ignore[reportUnusedFunction]
                if self._call is call:
                    await self.hang_up("hung up")

            @channel.on("message")
            def said(message: object) -> None:  # pyright: ignore[reportUnusedFunction]
                if self._call is not call:
                    # Queued before its call was let go: the call it spoke for is gone, and hands is elsewhere.
                    return
                call.heard.set()
                try:
                    self._hear(call, parse_said(message))
                except Rejected as error:
                    # [LAW:no-silent-failure] the page and hands disagree about what it sends: said, not guessed at.
                    logger.error(str(error))

            if self._offered is call:
                await self._arrive(call)

        @peer.on("connectionstatechange")
        async def changed() -> None:  # pyright: ignore[reportUnusedFunction]
            if peer.connectionState == "failed":
                if self._call is call:
                    await self.hang_up("failed")
                if self._offered is call:
                    self._offered = None
                    await self._let_go(call, "failed")

        try:
            await peer.setRemoteDescription(RTCSessionDescription(offer.sdp, offer.type))
            await peer.setLocalDescription(await peer.createAnswer())
        except BaseException as error:
            # Never offered, so nothing else ends its call or holds the peer to close it.
            call_ended(self._record, began, remote, error)
            await peer.close()
            raise
        older, self._offered = self._offered, call
        await self._let_go(older, "replaced")
        return peer.localDescription

    async def _arrive(self, call: _Call) -> None:
        # [LAW:no-ambient-temporal-coupling] the call that was up goes, and this one comes, in one step before any
        # await: a speaker reading the gate between them never finds the phone without a call.
        self._offered = None
        gone = self._leave("replaced")
        self._call, call.arrived_ms = call, since(call.began.began)
        # [LAW:nothing-unseen] the move a call makes is part of that call, in its trace, wherever the call's end is noticed.
        with continuing(call.began.span):
            self._key.go("phone")
        # Watched from the channel's opening: the page sends nothing before it.
        call.watch = asyncio.create_task(self._watch(call), name="the phone's call watch")
        match gone:
            case None:
                pass
            case _Call():
                await gone.peer.close()

    async def _let_go(self, offered: _Call | None, reason: PhoneGone) -> None:
        # An offer is let go of once it is no longer the one offered, so no other handler of it acts again.
        match offered:
            case None:
                pass
            case _Call():
                call_ended(self._record, offered.began, offered.remote, CallUnreached(reason))
                await offered.peer.close()

    async def _watch(self, call: _Call) -> None:
        while True:
            try:
                await asyncio.wait_for(call.heard.wait(), QUIET_SECS)
            except TimeoutError:
                if self._call is call:
                    await self.hang_up("went quiet")
                return
            call.heard.clear()

    def _hear(self, call: _Call, said: Said) -> None:
        match said:
            case bytes():
                frame = AudioFrame.from_ndarray(np.frombuffer(said, dtype=np.int16)[None, :], format="s16", layout="mono")
                frame.sample_rate = call.rate
                for out in call.resampler.resample(frame):
                    # One read of the gate per frame, as the desk's capture thread reads it.
                    gate = self._key.gate
                    if gate.hears("phone"):
                        audio = out.to_ndarray().astype(np.int16).tobytes()
                        self.heard.put_nowait(gate.framed(audio, audio, self._heard_rate, 1, "phone"))
            case move:
                match self._key.move(move, "phone button"):
                    case None:
                        pass
                    case taken:
                        self.moves.put_nowait(taken)

    def _leave(self, reason: PhoneGone) -> _Call | None:
        # [LAW:no-ambient-temporal-coupling] the place moves in the same step the call goes, before any await: a
        # speaker reading the gate after this never finds the phone without a call.
        call, self._call = self._call, None
        match call:
            case None:
                return None
            case _Call():
                with continuing(call.began.span):
                    self._key.go("desk")
                call.outbound.stop()
                if call.watch is not None and call.watch is not asyncio.current_task():
                    call.watch.cancel()
                call_ended(self._record, call.began, call.remote, CallLeft(reason, call.arrived_ms))
                return call

    async def hang_up(self, reason: PhoneGone) -> None:
        """End the call, if there is one, and be at the desk again."""
        match self._leave(reason):
            case None:
                pass
            case call:
                await call.peer.close()

    async def stop(self) -> None:
        """End the call and let go of any offer, as hands stops."""
        offered, self._offered = self._offered, None
        await self._let_go(offered, "stopped")
        await self.hang_up("stopped")

    def play(self, audio: bytes) -> asyncio.Future[None]:
        """Send 16-bit mono audio at the played rate to the phone; resolved once it has gone, or the call has ended."""
        match self._call:
            case None:
                raise RuntimeError("hands was told to play to the phone with no call up")
            case call:
                return call.outbound.give(audio)
