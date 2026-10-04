"""The system channel: what hands says about itself, and where it goes when speech or the model is what failed."""

import asyncio
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anthropic
import httpx2
import openai
import pytest
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.errors import ErrorCategory

from conftest import unprimed
from hands.sessions import heartbeat
from hands.daemon.cli import crashed_before
from hands.sessions.audit import Announced, Entry, HoldHeard, Levels
from hands.daemon import notify
from hands.daemon.notify import notification_command, post_notification
from hands.sessions.home import Home
from hands.voice.system import (
    NoMicrophone,
    TurnExpired,
    told,
    AudioMoved,
    BURST_SECONDS,
    Deaf,
    ModelFailed,
    ModelFault,
    UsageLimitReached,
    ModelReplyEmpty,
    ModelUnreachable,
    Post,
    Say,
    Started,
    SystemChannel,
    SystemFact,
    TranscriptionFailed,
    Unrouted,
    alarm,
    system_text,
)
from hands.voice import transcription
from hands.voice.microphone import Devices
from hands.voice.transcription import Broken, Busy, Loading, Lost, NotServing, ServerError, Unanswered, Unreachable
from hands.voice.hold import Move
from hands.voice.ptt import Gate
from hands.voice.turnstop import TurnResolved
from hands.voice.whisper import Whisper


BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
NOT_SERVING = "LowTalker is not serving transcription. Start LowTalker's network build and switch Serve Transcription on in its menu"
UNREACHABLE = "the transcription server's address cannot be reached. Check the transcription url in hands' config.toml"
LOADING = "LowTalker's model is still loading. Speak again once LowTalker's menu says the model is ready"
BROKEN = "LowTalker answered with something that is not a transcription. Check that the transcription url in hands' config.toml is LowTalker's"
DEAF = Devices(input=None, output="Mac mini Speakers")


@pytest.mark.parametrize(
    ("fact", "said"),
    [
        (Started(after_crash=False, devices=BUILT_IN), "hands is up."),
        (Started(after_crash=True, devices=BUILT_IN), "hands is back after a crash."),
        (Started(after_crash=False, devices=DEAF), "hands is up, but there is no microphone, so it cannot hear you."),
        (Started(after_crash=True, devices=DEAF), "hands is back after a crash, but there is no microphone, so it cannot hear you."),
        (Deaf(NotServing("refused")), f"hands cannot hear you: {NOT_SERVING}."),
        (Deaf(Unreachable("ClientConnectorDNSError: nodename nor servname provided")), f"hands cannot hear you: {UNREACHABLE}."),
        (Deaf(Loading("model not ready")), f"hands cannot hear you: {LOADING}."),
        (Deaf(Broken("404: Not Found")), f"hands cannot hear you: {BROKEN}."),
        (AudioMoved(DEAF), "No microphone: hands cannot hear you. Speaking on Mac mini Speakers."),
        (NoMicrophone(), "There is no microphone, so hands cannot hear you."),
        (ModelUnreachable(), "The language model is unreachable."),
        (ModelFailed(ErrorCategory.RATE_LIMIT), "The language model failed: rate limit."),
        (ModelReplyEmpty(), "The language model sent back nothing."),
        (UsageLimitReached(None), "The language model's usage limit is reached."),
        (TranscriptionFailed(None), "That turn was not heard: speech recognition failed."),
        (TranscriptionFailed(NotServing("refused")), f"That turn was not heard: {NOT_SERVING}."),
        (TranscriptionFailed(Unreachable("ClientConnectorCertificateError: certificate verify failed")), f"That turn was not heard: {UNREACHABLE}."),
        (TranscriptionFailed(Loading("model not ready")), f"That turn was not heard: {LOADING}."),
        (TranscriptionFailed(Busy("too many")), "That turn was not heard: LowTalker is already transcribing four things at once. Say it again in a moment."),
        (
            TranscriptionFailed(Unanswered(30.0)),
            "That turn was not heard: LowTalker did not answer within 30 seconds. Say it again, and restart LowTalker if it keeps happening.",
        ),
        (TranscriptionFailed(Lost("ServerDisconnectedError")), "That turn was not heard: LowTalker dropped the connection before it answered. Say it again."),
        (TranscriptionFailed(Broken("404: Not Found")), f"That turn was not heard: {BROKEN}."),
        (
            TranscriptionFailed(ServerError("500: decode failed")),
            "That turn was not heard: LowTalker failed on its side. Say it again, and restart LowTalker if it keeps happening.",
        ),
        (TurnExpired(), "That turn was open for 120 seconds, so hands threw it away."),
    ],
)
def test_each_fact_is_said_from_its_template(fact: SystemFact, said: str) -> None:
    assert system_text(fact) == said


@pytest.mark.parametrize(
    ("move", "devices", "said"),
    [
        ("start", DEAF, (NoMicrophone(),)),
        ("listen", DEAF, (NoMicrophone(),)),
        ("listen", BUILT_IN, ()),
        ("stop", DEAF, ()),
        ("drop", DEAF, ()),
        ("start", BUILT_IN, ()),
        ("expire", BUILT_IN, (TurnExpired(),)),
        ("expire", DEAF, ()),
    ],
)
def test_a_move_the_tone_alone_would_leave_unexplained_is_said(move: Move, devices: Devices, said: tuple[SystemFact, ...]) -> None:
    assert told(move, devices) == said


class Services:
    def __init__(self) -> None:
        self.stt, self.llm, self.tts, self.transport = FrameProcessor(), FrameProcessor(), FrameProcessor(), FrameProcessor()

    def alarm(self, error: ErrorFrame) -> object:
        return alarm(error, stt=self.stt, llm=self.llm, tts=self.tts)


REFUSED = openai.APIConnectionError(request=httpx2.Request("POST", "http://192.168.7.240:8080/v1/chat/completions"))

# What Anthropic answered every call with from 15:07 on 2026-09-27, which hands said as "invalid request".
LIMIT = "You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."
RETURNS = datetime(2026, 10, 1, tzinfo=UTC).timestamp()


def anthropic_error(status: int, message: str) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.APIStatusError(f"Error code: {status} - {message}", response=response, body={"type": "error", "error": {"type": "api_error", "message": message}})


def openai_streamed_error(message: str) -> openai.APIError:
    return openai.APIError(message, httpx2.Request("POST", "http://inferno.local:8080/v1/chat/completions"), body={"message": message})


@pytest.mark.parametrize(
    ("exception", "category", "fact"),
    [
        (anthropic_error(400, LIMIT), ErrorCategory.INVALID_REQUEST, UsageLimitReached(RETURNS)),
        (anthropic_error(400, "You have reached your specified API usage limits."), ErrorCategory.INVALID_REQUEST, UsageLimitReached(None)),
        (openai_streamed_error(LIMIT), ErrorCategory.UNKNOWN, UsageLimitReached(RETURNS)),
        # Anything else keeps its category, and none of the API's own text is said: it can carry ids, counts and URLs.
        (anthropic_error(400, "prompt is too long: 205113 tokens > 200000 maximum"), ErrorCategory.INVALID_REQUEST, ModelFailed(ErrorCategory.INVALID_REQUEST)),
        (anthropic_error(529, "Overloaded"), ErrorCategory.SERVER, ModelFailed(ErrorCategory.SERVER)),
        (RuntimeError("boom"), ErrorCategory.UNKNOWN, ModelFailed(ErrorCategory.UNKNOWN)),
        # Under the brain the stage has read the fact off the wire already, and it is said as read.
        (ModelFault(UsageLimitReached(RETURNS)), ErrorCategory.UNKNOWN, UsageLimitReached(RETURNS)),
        (ModelFault(ModelUnreachable()), ErrorCategory.UNKNOWN, ModelUnreachable()),
        (ModelFault(ModelReplyEmpty()), ErrorCategory.UNKNOWN, ModelReplyEmpty()),
    ],
)
def test_a_spent_usage_limit_is_said_as_itself_and_every_other_refusal_by_its_category(exception: Exception, category: ErrorCategory, fact: SystemFact) -> None:
    services = Services()
    error = ErrorFrame(f"Unknown error occurred: {exception}", exception=exception, processor=services.llm, category=category)
    assert services.alarm(error) == Say(fact)


def test_the_limit_is_said_to_lift_in_the_listeners_own_time() -> None:
    # The zone is the process's, so it is put back exactly as found, and re-read, before any other test runs.
    before = os.environ.get("TZ")
    os.environ["TZ"] = "America/Denver"
    time.tzset()
    try:
        assert system_text(UsageLimitReached(RETURNS)) == "The language model's usage limit is reached, until September 30 at 6:00 PM."
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


def test_an_error_is_told_by_the_processor_that_raised_it() -> None:
    services = Services()
    # A stopped model server refuses the connection; Pipecat files the SDK's error as UNKNOWN.
    assert services.alarm(ErrorFrame("Error during completion", exception=REFUSED, processor=services.llm, category=ErrorCategory.UNKNOWN)) == Say(ModelUnreachable())
    assert services.alarm(ErrorFrame("timed out", processor=services.llm, category=ErrorCategory.CONNECTIVITY)) == Say(ModelUnreachable())
    assert services.alarm(ErrorFrame("bad key", processor=services.llm, category=ErrorCategory.AUTHENTICATION)) == Say(ModelFailed(ErrorCategory.AUTHENTICATION))
    assert services.alarm(ErrorFrame("boom", processor=services.stt)) == Say(TranscriptionFailed(None))
    # The transcription server's fault is carried to what is said, never read back out of the error's text.
    loading = transcription.TranscriptionFailed("http://127.0.0.1:8610/v1", Loading("model not ready"))
    assert services.alarm(ErrorFrame("Whisper could not transcribe hold 1", exception=loading, processor=services.stt)) == Say(TranscriptionFailed(Loading("model not ready")))
    assert services.alarm(ErrorFrame("no voice", processor=services.tts)) == Post("no voice")
    # An error that names no processor is not the model's.
    assert services.alarm(ErrorFrame("lost")) == Unrouted("no processor", "lost")
    assert services.alarm(ErrorFrame("device gone", processor=services.transport)) == Unrouted(str(services.transport), "device gone")


class Recorder(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self.frames: list[Frame] = []

    async def queue_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM, callback: object = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        await asyncio.sleep(0)  # a real processor's queue is awaited, and the channel must hold across it
        self.frames.append(frame)


async def test_a_fact_goes_to_speech_while_it_works_and_to_the_screen_when_it_does_not() -> None:
    tts, posted = Recorder(), list[str]()

    async def notify(text: str) -> bool:
        posted.append(text)
        return True

    recorded: list[Entry] = []
    channel = SystemChannel(tts, notify, recorded.append)
    await channel.say(ModelUnreachable())
    # Kept out of the model's context: there it would read as the model's own reply.
    assert [(type(frame), getattr(frame, "text", None), getattr(frame, "append_to_context", None)) for frame in tts.frames] == [
        (TTSSpeakFrame, "The language model is unreachable.", False)
    ]
    await tts.set_usable(False)
    await channel.say(TranscriptionFailed(None))
    await channel.sound(Post("no voice"))
    assert len(tts.frames) == 1
    assert posted == ["hands cannot speak, so: That turn was not heard: speech recognition failed.", "hands cannot speak: no voice"]
    assert recorded == [
        Announced("The language model is unreachable.", "speech"),
        Announced("That turn was not heard: speech recognition failed.", "screen"),
        Announced("hands cannot speak: no voice", "screen"),
    ]


class Clock:
    """A clock the test winds, so crossing a burst window costs no wall time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def speaking(clock: Clock) -> tuple[Recorder, list[Entry], SystemChannel]:
    """A channel whose speech works, so nothing it says should reach the screen."""
    tts, recorded = Recorder(), list[Entry]()

    async def notify(_text: str) -> bool:
        raise AssertionError("speech works, so nothing goes to the screen")

    return tts, recorded, SystemChannel(tts, notify, recorded.append, clock)


def said(tts: Recorder) -> list[str | None]:
    return [getattr(frame, "text", None) for frame in tts.frames]


async def test_a_fact_repeated_through_one_burst_is_said_once() -> None:
    """A key held down on 2026-09-22 queued hundreds of empty turns, and this channel reported each one every 0.43 s
    for as long as they drained. A fault that recurs recurs in bursts, and a burst that fills the only channel the
    daemon has is not a loud failure but a jammed one."""
    clock = Clock()
    tts, recorded, channel = speaking(clock)
    for _ in range(200):
        await channel.say(TranscriptionFailed(None))
        clock.now += 0.43
    assert said(tts) == ["That turn was not heard: speech recognition failed."] * 9  # 200 turns over 86 s, not 200 sentences
    # The audit says a thing was announced only where it was: what a burst costs is the saying, not the knowing.
    assert recorded == [Announced("That turn was not heard: speech recognition failed.", "speech")] * 9


async def test_a_fault_that_is_still_happening_is_said_again_once_its_burst_has_passed() -> None:
    """Nothing else is ever said while a model is down, so only time can end the burst. Suppressing until
    something else was said would leave a user pressing a key at a daemon that has gone permanently silent."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    await channel.say(TranscriptionFailed(None))
    clock.now += BURST_SECONDS - 0.01
    await channel.say(TranscriptionFailed(None))
    assert said(tts) == ["That turn was not heard: speech recognition failed."]
    clock.now += 0.01
    await channel.say(TranscriptionFailed(None))
    assert said(tts) == ["That turn was not heard: speech recognition failed."] * 2


async def test_two_faults_taking_turns_do_not_between_them_defeat_the_burst() -> None:
    """Two faults can interleave, as a turn that fails and a press with no microphone. Were the window one slot
    wide, each would be news to the other and the pair would speak at the full jammed cadence."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (TranscriptionFailed(None), NoMicrophone()) * 20:
        await channel.say(fact)
        clock.now += 0.43
    assert said(tts) == [
        "That turn was not heard: speech recognition failed.",
        "There is no microphone, so hands cannot hear you.",
        "That turn was not heard: speech recognition failed.",
        "There is no microphone, so hands cannot hear you.",
    ]


async def test_two_model_failures_of_different_kinds_are_both_heard() -> None:
    """The facts carry what differs, so sameness is the type's answer and not a comparison of sentences."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.RATE_LIMIT)):
        await channel.say(fact)
    assert len(tts.frames) == 2
    for fact in (UsageLimitReached(RETURNS), UsageLimitReached(RETURNS)):
        await channel.say(fact)
    assert len(tts.frames) == 3


async def test_the_screen_is_not_filled_by_a_burst_either() -> None:
    """A TTS that fails raises one error frame per frame it was handed, and the screen jams exactly as the ear
    did — the same window, taken from the other path. Pipecat's text for a silent utterance carries a fresh
    context id each time, so a window taken on the sentence rather than on the fault would never once close."""
    clock, posted = Clock(), list[str]()

    async def notify(text: str) -> bool:
        posted.append(text)
        return True

    channel = SystemChannel(Recorder(), notify, lambda _: None, clock)
    for turn in range(200):
        await channel.sound(Post(f"TTS context 0000-{turn:04d} completed with no audio"))
        clock.now += 0.43
    assert len(posted) == 9  # 200 turns over 86 s, not 200 notifications
    assert posted[0] == "hands cannot speak: TTS context 0000-0000 completed with no audio"


async def test_a_post_the_screen_refused_is_not_taken_for_one_the_user_saw() -> None:
    """A run started over ssh may have no GUI session to post into, so osascript refuses every time. The refusal is no
    announcement and goes on no audit — and it still costs one attempt, because a screen that always says no is
    the one case where recording only what landed would hammer it at the jammed cadence forever."""
    clock, recorded, attempts = Clock(), list[Entry](), list[str]()

    async def notify(text: str) -> bool:
        attempts.append(text)
        return False

    tts = Recorder()
    await tts.set_usable(False)
    channel = SystemChannel(tts, notify, recorded.append, clock)
    for _ in range(200):
        await channel.say(TranscriptionFailed(None))
        clock.now += 0.43
    assert attempts == ["hands cannot speak, so: That turn was not heard: speech recognition failed."] * 9
    assert recorded == []


async def test_a_burst_that_arrives_all_at_once_is_still_said_once() -> None:
    """Pipecat registers `on_pipeline_error` async (`sync: bool = False`), so it dispatches each error on its own
    task and a dead TTS puts one of these in flight per queued frame. A window taken after the delivery rather
    than at the decision is a window that none of them would have seen, and the jam comes straight back."""
    posted, tts = list[str](), Recorder()

    async def notify(text: str) -> bool:
        await asyncio.sleep(0)  # the await they are all in flight across
        posted.append(text)
        return True

    channel = SystemChannel(tts, notify, lambda _: None, Clock())
    await asyncio.gather(*(channel.sound(Post(f"TTS context 0000-{turn:04d} completed with no audio")) for turn in range(200)))
    assert posted == ["hands cannot speak: TTS context 0000-0000 completed with no audio"]
    await asyncio.gather(*(channel.say(TranscriptionFailed(None)) for _ in range(200)))
    assert said(tts) == ["That turn was not heard: speech recognition failed."]


async def test_whisper_is_done_with_every_hold_and_says_nothing_of_one_it_heard_nothing_in(monkeypatch: pytest.MonkeyPatch) -> None:
    said: list[str | None | Exception] = []

    async def transcribe(_self: Whisper, hold: int, levels: Levels, _audio: bytes) -> HoldHeard:
        match said.pop():
            case Exception() as error:
                raise error
            case text:
                return HoldHeard(hold, text, (), levels)

    monkeypatch.setattr(Whisper, "_heard", transcribe)
    whisper = Whisper(url="http://unused/v1", prompt=unprimed, record=lambda _: None)
    # What the pipeline's start sets: a sent hold is wrapped as a WAV at this rate.
    whisper._sample_rate = 16_000  # pyright: ignore[reportPrivateUsage]

    async def push(_frame: Frame, _direction: object = None) -> None:
        pass

    monkeypatch.setattr(whisper, "push_frame", push)
    # Three turns sent, one for each transcription below.
    gate = Gate()
    for _ in range(3):
        for gate in (gate.after("start", "desk"), gate.after("start", "desk").after("stop", "desk")):
            await whisper.process_audio_frame(gate.framed(b"\x00\x00", b"\x00\x00", 16000, 1, "desk"), FrameDirection.DOWNSTREAM)
    # Every transcription ends with Whisper done with its hold, and one it heard nothing in yields nothing else: no
    # frame that could reach the speaker. (Every frame has an id of its own, so frames made here are told by their kind.)
    said.append(None)
    assert [type(frame) async for frame in whisper.run_stt(b"")] == [TurnResolved]
    said.append("what time is it")
    assert [type(frame) async for frame in whisper.run_stt(b"")] == [TranscriptionFrame, TurnResolved]
    said.append(RuntimeError("model failed"))
    assert [type(frame) async for frame in whisper.run_stt(b"")] == [ErrorFrame, TurnResolved]


def test_the_notification_text_is_an_argument_not_part_of_the_script() -> None:
    command = notification_command('-say "hi" \\ now')
    assert command[-2:] == ["--", '-say "hi" \\ now']
    assert all('"hi"' not in part for part in command[:-1])


async def test_a_notification_osascript_never_posts_is_refused_and_said(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stuck(*_argv: str, timeout: float) -> None:
        raise TimeoutError

    monkeypatch.setattr(notify, "run", stuck)
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    try:
        assert await post_notification("hello") is False
    finally:
        logger.remove(sink)
    assert errors == [f"osascript did not post 'hello' in {notify.POST_TIMEOUT_SECONDS:.0f}s"]


GONE = "a pid no process holds"


@pytest.mark.parametrize(
    ("pipeline", "pid", "written_ago", "crashed"),
    [
        (None, GONE, 0, False),  # never ran
        ("stopped", GONE, 0, False),
        ("running", GONE, 0, True),
        ("starting", GONE, 0, True),
        ("running", os.getpid(), 0, False),  # up: this is not a restart
        ("running", os.getpid(), 60, True),  # hung, and something now holds its pid
    ],
)
def test_a_run_crashed_when_its_last_heartbeat_was_not_a_stop_and_it_is_not_beating(
    pipeline: heartbeat.PipelineState | None, pid: int | str, written_ago: int, crashed: bool, tmp_path: Path, dead_pid: Callable[[], int]
) -> None:
    home = Home(tmp_path)
    now = datetime.now(UTC)
    if pipeline is not None:
        heartbeat.write(home.status, heartbeat.Status(dead_pid() if pid == GONE else int(pid), now, now - timedelta(seconds=written_ago), heartbeat.HEARTBEAT, pipeline, None, 0, False, False))
    assert crashed_before(home) is crashed


def test_a_heartbeat_whose_pid_a_later_process_took_is_a_crash(tmp_path: Path) -> None:
    home = Home(tmp_path)
    day_ago = datetime.now(UTC) - timedelta(days=1)
    # This process holds the pid, but it started long after the run that wrote the heartbeat.
    heartbeat.write(home.status, heartbeat.Status(os.getpid(), day_ago, day_ago, heartbeat.HEARTBEAT, "running", None, 0, False, False))
    assert crashed_before(home) is True


def test_a_heartbeat_that_does_not_parse_is_not_taken_for_a_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.status.write_text("{")
    assert crashed_before(home) is False
    assert "not counted as a crash" in capsys.readouterr().err
