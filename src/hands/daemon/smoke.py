"""`hands smoke`: three spoken turns through the hands that is running, each part of the pipeline shown to have done its
share, and the first part that did not named.

The test calls hands as the phone's page does, so no microphone, talk key, or key event of the machine's is touched, and
says its turns in a voice macOS synthesizes. Its working session is a `claude` started as a person at a terminal starts
one, from whatever `claude` is first on the PATH, in a folder of the home's own that holds one file, named for a word
picked at random. The user asks hands to have that session say the file's name, sends the draft hands reads back, and
asks what the session said: the word comes back aloud only if every part did its share, since nothing but the session
could have read it.

Each stage is read off what the pipeline itself leaves: the membership the session's hooks write, the audit lines the
daemon writes as it hears a hold, types into a session, and takes a session's Stop, and the speech the call carries
back, transcribed by the server hands transcribes with. What hands said is judged from that speech alone, whichever
backend said it and whether in its own words or a tool's readback: it is what the user would have heard.
"""

import asyncio
import io
import json
import os
import random
import shutil
import ssl
import tempfile
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

from hands.core.session import Membership, SessionId
from hands.daemon.config import load
from hands.sessions import audit, heartbeat
from hands.sessions.child import run
from hands.sessions.home import Home
from hands.sessions.membership import parse_membership
from hands.sessions.payload import Rejected
from hands.sessions.pseudoterminal import ClaudeCode, on_terminal
from hands.sessions.untap import untapped
from hands.sessions.wide import annotate
from hands.voice import transcription
from hands.voice.phonepage import PHONE_PORT, phone_key

# Each stage of the pipeline, in the order the turns reach it: hands running; a session started from the `claude` on
# PATH joined to hands under fritter; the call up; the user's words heard as text; hands answering aloud about the
# session; the draft typed into the session; the session's turn finished; and what the session said told back aloud.
Stage = Literal["up", "joined", "called", "heard", "answered", "typed", "finished", "told"]
# The stages read off the audit log, one record each; hands telling the user of the session's turn is part of telling
# back what the session said.
Logged = Literal["heard", "typed", "finished", "told"]

# What the user says, in order: the request, the go-ahead for the draft hands reads back, and the question whose answer
# carries the word. The session is named by its project, the folder it runs in, as hands names every session.
FOLDER = "smoke"
SAID = (
    f"Tell the {FOLDER} session to reply with only the name of the one file in its folder.",
    "Send it.",
    f"What did the {FOLDER} session say?",
)
# What a session gives each process it runs, naming itself as their parent: Claude Code's (2.1.288), and fritter's
# address. A `claude` started under Claude Code's is a child of that session, not one of its own: it writes its turns
# into the parent's transcript, and its own is never made, so hands has nothing to read; and one started outside hands'
# shim under fritter's would join hands as that session's. `hands smoke` run from a session's shell starts its session
# as from a terminal outside one.
SESSION_GIVEN = (
    "FRITTER_SOCKET", "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_TMUX_TRUECOLOR",
)
# Words a recogniser and a synthesized voice agree on, each with one spelling (a recogniser wrote "harbor" as "Harbour"),
# none of them a word anyone says to hands unprompted.
WORDS = ("marigold", "lantern", "falcon", "violin", "glacier", "pepper", "walrus", "saffron")

# How long each stage has. A working session's turn is a model's turn and a tool call; every other stage is hands' own.
JOIN_SECONDS = 60.0
CALL_SECONDS = 15.0
TURN_SECONDS = 60.0
FINISH_SECONDS = 180.0
TRANSCRIBE_SECONDS = 30.0
# How often the records are read again while a stage is waited on.
POLL_SECONDS = 0.2

# The call's audio, both ways: 16-bit mono at the rate `say` is asked for and Whisper hears at, in the page's 20 ms frames.
RATE = 16000
FRAME_SECS = 0.02
FRAME_BYTES = round(RATE * FRAME_SECS) * 2
# A frame of hands' speech is voiced above this RMS; the silence it sends between utterances is all zeros.
VOICED_RMS = 100.0
# hands has finished saying something once the call has carried nothing voiced for this long: longer than the pause
# between two of its sentences.
QUIET_SECS = 2.5


class NotReached(Exception):
    """A stage the pipeline did not reach, and what was seen instead."""

    def __init__(self, stage: Stage, why: str) -> None:
        super().__init__(f"{stage}: {why}")
        self.stage: Stage = stage
        self.why = why


type Line = Mapping[str, object]


def parsed(lines: Sequence[str]) -> list[Line]:
    """The audit lines as JSON; a torn line, which the log can hold after a write failed partway, is no evidence of anything."""
    kept: list[Line] = []
    for line in lines:
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            kept.append(value)  # pyright: ignore[reportUnknownArgumentType]  (a JSON object's keys are strings)
    return kept


def proof(stage: Logged, lines: Sequence[Line], session: SessionId) -> str | None:
    """What in the audit lines of one turn shows `stage` reached, or None while nothing does; raises NotReached where a
    line shows it never will be."""
    # [LAW:dataflow-not-control-flow] one reading per stage, of the records the daemon writes as it reaches it, in order.
    # A send's Typing line is written before it is typed: it was typed once the unit of work that typed it has ended
    # with no TypingFailed line for it. A telling's Refocused line is written as it is taken, before it is said: it was
    # said once the reply that follows it is, which is recorded once it has been played.
    sending: tuple[str, str] | None = None
    telling = False
    for line in lines:
        match stage, line:
            case "heard", {"type": "HoldHeard", "said": str(said)}:
                return f"heard {said!r}"
            case "heard", {"type": "HoldHeard", "said": None, "dropped": list(dropped)}:  # pyright: ignore[reportUnknownVariableType]  (counted, never read)
                raise NotReached("heard", f"Whisper took no words from the hold; it dropped {len(dropped)} segment(s)")  # pyright: ignore[reportUnknownArgumentType]
            case "typed", {"type": "Typing", "effect": {"session": str(typed), "input": {"prompt": str(prompt)}}, "span": {"span_id": str(span)}} if typed == session:
                sending = (span, prompt)
            case "typed", {"type": "TypingFailed", "effect": {"session": str(typed)}, "reason": str(reason)} if typed == session:
                raise NotReached("typed", f"hands could not type into session {session}: {reason}")
            case "typed", {"type": "WideEvent", "span_id": str(span)} if sending is not None and span == sending[0]:
                return f"typed {sending[1]!r} into session {session}"
            case "finished", {"type": "WideEvent", "event": "hook", "facts": {"hook": "Stop", "session": str(stopped)}} if stopped == session:
                return f"session {session} finished its turn"
            # How the finished turn reaches the user, which hands decides at once: held until asked, or told unasked.
            case "told", {"type": "WideEvent", "event": "utterance", "outcome": "failed", "error": str(error), "facts": {"session": str(told), "heard": {"type": "Summarise"}}} if told == session:
                raise NotReached("told", f"hands could not tell session {session}'s turn: {error}")
            case "told", {"type": "WideEvent", "event": "utterance", "facts": {"session": str(told), "heard": {"type": "Summarise"}, "delivered": {"type": "Withheld", "why": str(why)}}} if told == session:
                return f"hands holds session {session}'s turn until asked ({why})"
            case "told", {"type": "Refocused", "session": str(told)} if told == session:
                telling = True
            case "told", {"type": "Replied", "text": str(text)} if telling:
                return f"hands told the user of session {session}'s turn: {text!r}"
            case _:
                pass
    return None


def errors(lines: Sequence[Line]) -> list[str]:
    """Every line of the run the log marks as an error, as its type and what it says: what the daemon said went wrong."""
    return [
        f"{line.get('type')}: {line.get('message') or line.get('error') or line.get('reason') or line.get('event')}"
        for line in lines
        if line.get("level") == "error"
    ]


def joined(home: Home, folder: Path, before: frozenset[str]) -> Membership | None:
    """The membership a session started in `folder` since `before` was listed wrote at its first hook, if one has."""
    for path in sorted(home.memberships.glob("*.json")):
        if path.stem in before:
            continue
        try:
            membership = parse_membership(SessionId(path.stem), path.read_bytes())
        except (OSError, Rejected):
            # Being written, or already removed: read again at the next poll.
            continue
        if membership.cwd.resolve() == folder:
            return membership
    return None


def as_from_a_terminal(environment: Mapping[str, str], home: Home) -> dict[str, str]:
    """`environment` as a terminal outside any session has it, the home's sessions report to named: no tap and nothing
    else of a session the test may have been run inside."""
    return {**{name: value for name, value in untapped(environment).items() if name not in SESSION_GIVEN}, "HANDS_HOME": str(home.root)}


def as_wav(audio: bytes) -> bytes:
    """16-bit mono audio at RATE, as the WAV file a transcription server is sent."""
    written = io.BytesIO()
    with wave.open(written, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(audio)
    return written.getvalue()


@dataclass
class Ear:
    """The audio hands sends the call, as it arrives: each frame at RATE, mono, with when it came and whether it is voiced."""

    frames: list[tuple[float, bytes]] = field(default_factory=list[tuple[float, bytes]])
    last_voiced: float | None = None

    def heard(self, audio: bytes, at: float) -> None:
        self.frames.append((at, audio))
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
        if samples.size and float(np.sqrt(np.mean(samples**2))) > VOICED_RMS:
            self.last_voiced = at

    def since(self, at: float) -> bytes:
        return b"".join(audio for came, audio in self.frames if came >= at)

    def finished_speaking(self, since: float, now: float) -> float | None:
        """When hands last spoke, once it has spoken since `since` and gone quiet; None while it has not, or still speaks."""
        last = self.last_voiced
        return last if last is not None and last >= since and now - last >= QUIET_SECS else None


@dataclass
class Caller:
    """The test's end of a call to hands: what it says and presses, in order, at the page's pace, and what it hears."""

    peer: RTCPeerConnection
    channel: RTCDataChannel
    ear: Ear = field(default_factory=Ear)
    # [LAW:no-ambient-temporal-coupling] the button and the audio go out over one ordered channel, from one queue, so a
    # release follows every frame said before it, as the page's does.
    pending: deque[bytes | str] = field(default_factory=deque[bytes | str])
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    # Its voice and its ear, each running for as long as the call is up.
    tasks: list[asyncio.Task[None]] = field(default_factory=list[asyncio.Task[None]])

    def up(self, stage: Stage) -> None:
        """Raises NotReached at `stage` once the call's voice or ear has stopped: the call dropped, or broke."""
        # [LAW:no-silent-failure] what stopped either one is why `stage`, which needs the call, was not reached.
        for task in self.tasks:
            if task.done():
                stopped = None if task.cancelled() else task.exception()
                raise NotReached(stage, f"{task.get_name()} stopped: {stopped!r}" if stopped else f"{task.get_name()} stopped: the call dropped")

    async def keep_sending(self) -> None:
        """A frame every FRAME_SECS for as long as the call is up, silence where nothing is being said: hands hangs up a
        call that goes quiet, as it does a page that is closed."""
        loop = asyncio.get_running_loop()
        start, sent = loop.time(), 0
        silence = bytes(FRAME_BYTES)
        while True:
            said = self.pending.popleft() if self.pending else silence
            self.channel.send(said)
            if not self.pending:
                self.drained.set()
            match said:
                case bytes():
                    # Paced by the count sent against the clock at the first, as hands paces what it sends the page.
                    sent += 1
                    await asyncio.sleep(max(0.0, start + sent * FRAME_SECS - loop.time()))
                case str():
                    # A press or a release goes out with the frame after it, as the page's button does.
                    pass

    async def say(self, stage: Stage, audio: bytes) -> float:
        """Hold the button, say `audio`, and let go once it has all gone; when it was let go of, by the loop's clock.
        Raises NotReached at `stage` where the call stops before it has."""
        self.drained.clear()
        self.pending.extend(["press", *(audio[at : at + FRAME_BYTES] for at in range(0, len(audio), FRAME_BYTES)), "release"])
        drained = asyncio.create_task(self.drained.wait())
        try:
            await asyncio.wait([drained, *self.tasks], return_when=asyncio.FIRST_COMPLETED)
        finally:
            drained.cancel()
        self.up(stage)
        return asyncio.get_running_loop().time()

    async def keep_hearing(self, track: MediaStreamTrack) -> None:
        loop = asyncio.get_running_loop()
        resampler = AudioResampler(format="s16", layout="mono", rate=RATE)
        while True:
            try:
                frame = await track.recv()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  (typed for every kind of track at once)
            except MediaStreamError:
                return
            if not isinstance(frame, AudioFrame):
                raise TypeError("hands' audio track carried something that is not audio")
            for out in resampler.resample(frame):
                self.ear.heard(out.to_ndarray().astype(np.int16).tobytes(), loop.time())


@dataclass
class Run:
    """A smoke run as it goes: the audit lines it has read, and when it began."""

    home: Home
    word: str
    transcription: str
    offset: int
    lines: list[Line] = field(default_factory=list[Line])
    began: float = field(default_factory=time.monotonic)

    def read(self) -> None:
        """The audit lines the daemon has written since the last read."""
        fresh, self.offset = audit.past(self.home.audit, self.offset)
        self.lines.extend(parsed(fresh))

    def mark(self) -> int:
        """Where the lines of a turn about to begin start: past every line written before it."""
        self.read()
        return len(self.lines)

    def reached(self, stage: Stage, shown: str) -> None:
        # [LAW:nothing-unseen] when each stage was reached, from the start of the run, on the command's event.
        annotate(**{f"{stage}_ms": round((time.monotonic() - self.began) * 1000), "reached": stage})
        print(f"ok {stage}: {shown}", flush=True)


async def until[T](stage: Stage, seconds: float, found: Callable[[], Awaitable[T | None]], why: Callable[[], str]) -> T:
    """What `found` finds, once it finds it within `seconds`; raises NotReached with `why` when it does not."""
    deadline = time.monotonic() + seconds
    while (seen := await found()) is None:
        if time.monotonic() > deadline:
            raise NotReached(stage, why())
        await asyncio.sleep(POLL_SECONDS)
    return seen


async def smoke(home: Home, environment: Mapping[str, str]) -> int:
    """Run the test, printing each stage as it is reached; 0 once every stage is, 1 at the first that is not."""
    word = random.choice(WORDS)
    folder = (home.root / FOLDER).resolve()
    annotate(word=word, folder=folder)
    _, offset = audit.tail(home.audit, 0)
    try:
        # The server hands transcribes with, as the settings hands runs on name it.
        smoked = Run(home, word, load(home).config.transcription, offset)
    except Rejected as error:
        annotate(failed_at="up", why=f"hands cannot read its settings: {error}")
        print(f"FAILED up: hands cannot read its settings: {error}", flush=True)
        return 1
    annotate(transcription=smoked.transcription)
    try:
        await _stages(smoked, folder, environment)
    except NotReached as missed:
        smoked.read()
        said = errors(smoked.lines)
        # [LAW:nothing-unseen] why, as a fact: the command's failure is its exit code, which the CLI's layer sets after this.
        annotate(failed_at=missed.stage, why=missed.why, daemon_errors=tuple(said))
        print(f"FAILED {missed.stage}: {missed.why}", flush=True)
        for error in said:
            print(f"  hands logged: {error}", flush=True)
        return 1
    print("passed: every part of the pipeline did its share.", flush=True)
    return 0


async def _stages(smoked: Run, folder: Path, environment: Mapping[str, str]) -> None:
    home = smoked.home
    verdict = heartbeat.look(home.status, datetime.now(UTC))
    match verdict:
        case heartbeat.Up():
            smoked.reached("up", heartbeat.describe(verdict, datetime.now(UTC)))
        case _:
            raise NotReached("up", f"{heartbeat.describe(verdict, datetime.now(UTC))}; start it with `hands run`")
    claude = shutil.which("claude", path=environment.get("PATH"))
    if claude is None:
        raise NotReached("joined", "there is no `claude` on PATH")
    said = [await _synthesized(text) for text in SAID]
    # The folder holds the one file, named for this run's word, and nothing else.
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / f"{smoked.word}.txt").write_text("")
    before = frozenset(path.stem for path in home.memberships.glob("*.json"))
    session = await on_terminal([claude], folder, as_from_a_terminal(environment, home))
    try:
        member = await _joined(smoked, folder, before, claude, session)
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        caller = Caller(peer, peer.createDataChannel("talk", ordered=True))
        try:
            await _called(home, caller)
            smoked.reached("called", f"hands answered a call at https://127.0.0.1:{PHONE_PORT}")
            await _turns(smoked, caller, member.id, said)
        except NotReached as missed:
            # [LAW:nothing-unseen] what the session showed is part of why a stage it took part in was not reached.
            raise NotReached(missed.stage, f"{missed.why}\nthe smoke session showed:\n{session.shown()}") from missed
        finally:
            for task in caller.tasks:
                task.cancel()
            await peer.close()
    finally:
        await session.stop()


async def _joined(smoked: Run, folder: Path, before: frozenset[str], claude: str, session: ClaudeCode) -> Membership:
    home = smoked.home

    async def found() -> Membership | None:
        return joined(home, folder, before)

    member = await until(
        "joined", JOIN_SECONDS, found,
        lambda: f"`{claude}` started in {folder} never joined hands: no hook of its wrote a membership to {home.memberships}; it showed:\n{session.shown()}",
    )
    annotate(session=member.id)
    if member.fritter is None:
        raise NotReached("joined", f"session {member.id} joined hands, but not under fritter, so hands cannot type into it: `{claude}` is not hands' shim; put {home.bin} first on PATH")
    smoked.reached("joined", f"session {member.id}, under fritter, from `{claude}`")
    return member


async def _called(home: Home, caller: Caller) -> None:
    """The call up, as the page puts it up: its offer answered with the phone's key, and its channel open."""
    opened = asyncio.Event()
    caller.channel.on("open", opened.set)
    caller.peer.addTransceiver("audio", direction="recvonly")

    @caller.peer.on("track")
    def heard(track: MediaStreamTrack) -> None:  # pyright: ignore[reportUnusedFunction]
        caller.tasks.append(asyncio.create_task(caller.keep_hearing(track), name="the smoke test's ear"))

    await caller.peer.setLocalDescription(await caller.peer.createOffer())
    try:
        # hands' own certificate, which names 127.0.0.1, is the one authority this call trusts.
        trusted = ssl.create_default_context(cafile=home.phone / "own.crt")
        async with aiohttp.ClientSession() as client, client.post(
            f"https://127.0.0.1:{PHONE_PORT}/offer",
            json={"sdp": caller.peer.localDescription.sdp, "type": "offer", "rate": RATE},
            headers={"Authorization": f"Bearer {phone_key(home)}"},
            ssl=trusted,
        ) as answered:
            if answered.status != 200:
                raise NotReached("called", f"hands refused the call ({answered.status}): {await answered.text()}")
            answer: object = await answered.json()
    except (OSError, Rejected, aiohttp.ClientError, json.JSONDecodeError) as error:
        raise NotReached("called", f"hands' phone page at port {PHONE_PORT} could not be called: {error}") from error
    match answer:
        case {"sdp": str(sdp), "type": "answer"}:
            await caller.peer.setRemoteDescription(RTCSessionDescription(sdp, "answer"))
        case other:
            raise NotReached("called", f"hands answered the offer with {other!r}, which is no answer")
    try:
        async with asyncio.timeout(CALL_SECONDS):
            await opened.wait()
    except TimeoutError:
        raise NotReached("called", f"the call's channel did not open within {CALL_SECONDS:.0f}s; the connection is {caller.peer.connectionState}") from None
    caller.tasks.append(asyncio.create_task(caller.keep_sending(), name="the smoke test's voice"))


async def _turns(smoked: Run, caller: Caller, session: SessionId, said: Sequence[bytes]) -> None:
    asking, sending, questioning = said
    loop = asyncio.get_running_loop()

    def logged(stage: Logged, since: int, what: str) -> Callable[[], Awaitable[str]]:
        async def found() -> str | None:
            smoked.read()
            return proof(stage, smoked.lines[since:], session)

        return lambda: until(stage, FINISH_SECONDS if stage == "finished" else TURN_SECONDS, found, lambda: f"no {what} in the audit log in the time allowed")

    async def told_back(stage: Stage, released: float, holds: str) -> str:
        """What hands said back after `released`, transcribed, once it holds the word `holds`."""
        heard: list[str] = []
        transcribed_to: list[float] = []

        async def found() -> str | None:
            caller.up(stage)
            last = caller.ear.finished_speaking(released, loop.time())
            if last is None or last in transcribed_to:
                return None
            transcribed_to.append(last)
            heard[:] = [await _transcribed(stage, smoked.transcription, caller.ear.since(released))]
            return f"hands said {heard[0]!r}" if holds in heard[0].casefold() else None

        return await until(stage, TURN_SECONDS, found, lambda: f"what hands said back never named {holds!r}: it said {heard[0]!r}" if heard else "hands said nothing back on the call")

    # The request: heard, and hands answering aloud about the session.
    since = smoked.mark()
    released = await caller.say("heard", asking)
    smoked.reached("heard", await logged("heard", since, "transcription of the hold (HoldHeard)")())
    smoked.reached("answered", await told_back("answered", released, FOLDER))

    # The go-ahead: the draft typed into the session, and the session's turn run to its end.
    since = smoked.mark()
    released = await caller.say("typed", sending)
    smoked.reached("typed", await logged("typed", since, f"send to session {session} (Typing)")())
    smoked.reached("finished", await logged("finished", since, f"Stop hook from session {session}")())

    async def quiet() -> bool | None:
        return True if caller.ear.last_voiced is None or loop.time() - caller.ear.last_voiced >= QUIET_SECS else None

    # [LAW:no-ambient-temporal-coupling] hands may tell the user of the session's finished turn unasked; the question waits
    # for the record that it was held or has been told, then for its speech to end, so the question cuts nothing off and
    # is what is answered.
    held_or_told = await logged("told", since, f"telling of session {session}'s turn, nor its holding (utterance, Refocused)")()
    print(f"   {held_or_told}", flush=True)
    await until("told", TURN_SECONDS, quiet, lambda: "hands never stopped speaking after telling of the session's turn")

    # The question: what the session said, told back aloud.
    released = await caller.say("told", questioning)
    smoked.reached("told", await told_back("told", released, smoked.word))


async def _transcribed(stage: Stage, url: str, audio: bytes) -> str:
    """`audio`, as the server hands transcribes with hears it."""
    try:
        heard = await transcription.segments(url, as_wav(audio), "smoke.wav", None, TRANSCRIBE_SECONDS)
    except (transcription.TranscriptionFailed, aiohttp.ClientError, TimeoutError, OSError) as error:
        raise NotReached(stage, f"what hands said back could not be transcribed at {url}: {error}") from error
    return " ".join(segment.text for segment in heard).strip()


async def _synthesized(text: str) -> bytes:
    """`text` in macOS's own voice, as 16-bit mono at RATE."""
    with tempfile.TemporaryDirectory(prefix="hands-smoke-") as directory:
        spoken = Path(directory) / "said.wav"
        # The test's own voice, no part of hands: its failure is the test's, raised, never named as a stage of the pipeline.
        ran = await run("say", "-o", str(spoken), f"--data-format=LEI16@{RATE}", text, timeout=30.0)
        if ran.returncode != 0:
            raise RuntimeError(f"`say` could not synthesize what the test says ({ran.returncode}): {ran.err.decode(errors='replace').strip()}")
        with wave.open(str(spoken), "rb") as read:
            return read.readframes(read.getnframes())


def run_smoke(home: Home) -> int:
    """`hands smoke`, as the CLI runs it."""
    return asyncio.run(smoke(home, os.environ))
