"""The system channel: what hands says about itself, and where it goes when speech or the model is what failed."""

import asyncio
import os
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx2
import openai
import pytest
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.utils.errors import ErrorCategory

from hands.daemon import status
from hands.daemon.cli import crashed_before
from hands.sessions.audit import Announced, Entry
from hands.daemon.notify import notification_command
from hands.sessions.home import Home
from hands.voice.system import (
    AudioMoved,
    BURST_SECONDS,
    ModelFailed,
    ModelUnreachable,
    NothingTranscribed,
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
from hands.voice.microphone import Devices
from hands.voice.whisper import NOTHING_TRANSCRIBED, Whisper


BUILT_IN = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
DEAF = Devices(input=None, output="Mac mini Speakers")


@pytest.mark.parametrize(
    ("fact", "said"),
    [
        (Started(after_crash=False, devices=BUILT_IN), "hands is up."),
        (Started(after_crash=True, devices=BUILT_IN), "hands is back after a crash."),
        (Started(after_crash=False, devices=DEAF), "hands is up, but there is no microphone, so it cannot hear you."),
        (Started(after_crash=True, devices=DEAF), "hands is back after a crash, but there is no microphone, so it cannot hear you."),
        (AudioMoved(DEAF), "No microphone: hands cannot hear you. Speaking on Mac mini Speakers."),
        (ModelUnreachable(), "The language model is unreachable."),
        (ModelFailed(ErrorCategory.RATE_LIMIT), "The language model failed: rate limit."),
        (TranscriptionFailed(), "Speech recognition failed for that turn."),
        (NothingTranscribed(), "Whisper returned nothing for that turn."),
    ],
)
def test_each_fact_is_said_from_its_template(fact: SystemFact, said: str) -> None:
    assert system_text(fact) == said


class Services:
    def __init__(self) -> None:
        self.stt, self.llm, self.tts, self.transport = FrameProcessor(), FrameProcessor(), FrameProcessor(), FrameProcessor()

    def alarm(self, error: ErrorFrame) -> object:
        return alarm(error, stt=self.stt, llm=self.llm, tts=self.tts)


REFUSED = openai.APIConnectionError(request=httpx2.Request("POST", "http://192.168.7.240:8080/v1/chat/completions"))


def test_an_error_is_told_by_the_processor_that_raised_it() -> None:
    services = Services()
    # A stopped model server refuses the connection; Pipecat files the SDK's error as UNKNOWN.
    assert services.alarm(ErrorFrame("Error during completion", exception=REFUSED, processor=services.llm, category=ErrorCategory.UNKNOWN)) == Say(ModelUnreachable())
    assert services.alarm(ErrorFrame("timed out", processor=services.llm, category=ErrorCategory.CONNECTIVITY)) == Say(ModelUnreachable())
    assert services.alarm(ErrorFrame("bad key", processor=services.llm, category=ErrorCategory.AUTHENTICATION)) == Say(ModelFailed(ErrorCategory.AUTHENTICATION))
    assert services.alarm(ErrorFrame("boom", processor=services.stt)) == Say(TranscriptionFailed())
    assert services.alarm(ErrorFrame("no voice", processor=services.tts)) == Post("no voice")
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
    await channel.say(NothingTranscribed())
    await channel.sound(Post("no voice"))
    assert len(tts.frames) == 1
    assert posted == ["hands cannot speak, so: Whisper returned nothing for that turn.", "hands cannot speak: no voice"]
    assert recorded == [
        Announced("The language model is unreachable.", "speech"),
        Announced("Whisper returned nothing for that turn.", "screen"),
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
    """A key held down on 2026-09-22 queued hundreds of empty turns, and this channel said "Whisper returned
    nothing for that turn." every 0.43 s for as long as they drained. A fault that recurs recurs in bursts, and a
    burst that fills the only channel the daemon has is not a loud failure but a jammed one."""
    clock = Clock()
    tts, recorded, channel = speaking(clock)
    for _ in range(200):
        await channel.say(NothingTranscribed())
        clock.now += 0.43
    assert said(tts) == ["Whisper returned nothing for that turn."] * 9  # 200 turns over 86 s, not 200 sentences
    # The audit says a thing was announced only where it was: what a burst costs is the saying, not the knowing.
    assert recorded == [Announced("Whisper returned nothing for that turn.", "speech")] * 9


async def test_a_fault_that_is_still_happening_is_said_again_once_its_burst_has_passed() -> None:
    """Nothing else is ever said while the microphone is muted, so only time can end the burst. Suppressing until
    something else was said would leave a user pressing a key at a daemon that has gone permanently silent."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    await channel.say(NothingTranscribed())
    clock.now += BURST_SECONDS - 0.01
    await channel.say(NothingTranscribed())
    assert said(tts) == ["Whisper returned nothing for that turn."]
    clock.now += 0.01
    await channel.say(NothingTranscribed())
    assert said(tts) == ["Whisper returned nothing for that turn."] * 2


async def test_two_faults_taking_turns_do_not_between_them_defeat_the_burst() -> None:
    """The drain interleaves: some queued turns transcribe to nothing and some raise. Were the window one slot
    wide, each would be news to the other and the pair would speak at the full jammed cadence."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (NothingTranscribed(), TranscriptionFailed()) * 20:
        await channel.say(fact)
        clock.now += 0.43
    assert said(tts) == [
        "Whisper returned nothing for that turn.",
        "Speech recognition failed for that turn.",
        "Whisper returned nothing for that turn.",
        "Speech recognition failed for that turn.",
    ]


async def test_two_model_failures_of_different_kinds_are_both_heard() -> None:
    """The facts carry what differs, so sameness is the type's answer and not a comparison of sentences."""
    clock = Clock()
    tts, _, channel = speaking(clock)
    for fact in (ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.CONNECTIVITY), ModelFailed(ErrorCategory.RATE_LIMIT)):
        await channel.say(fact)
    assert len(tts.frames) == 2


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
    """Under launchd there may be no GUI session to post into, so osascript refuses every time. The refusal is no
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
        await channel.say(NothingTranscribed())
        clock.now += 0.43
    assert attempts == ["hands cannot speak, so: Whisper returned nothing for that turn."] * 9
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
    await asyncio.gather(*(channel.say(NothingTranscribed()) for _ in range(200)))
    assert said(tts) == ["Whisper returned nothing for that turn."]


async def test_whisper_reports_a_turn_it_transcribed_to_nothing_and_only_that(monkeypatch: pytest.MonkeyPatch) -> None:
    yielded: list[Frame] = []

    async def transcribe(_self: WhisperSTTServiceMLX, _audio: bytes) -> AsyncGenerator[Frame, None]:
        for frame in yielded:
            yield frame

    monkeypatch.setattr(WhisperSTTServiceMLX, "run_stt", transcribe)
    whisper = Whisper(settings=WhisperSTTServiceMLX.Settings(model="unused"))
    reports: list[None] = []

    @whisper.event_handler(NOTHING_TRANSCRIBED)
    async def empty(_stt: Whisper) -> None:  # pyright: ignore[reportUnusedFunction]
        reports.append(None)

    assert [frame async for frame in whisper.run_stt(b"")] == []
    yielded.append(TranscriptionFrame("what time is it", "user", "now"))
    assert [frame async for frame in whisper.run_stt(b"")] == yielded
    yielded[:] = [ErrorFrame("model failed")]
    assert [frame async for frame in whisper.run_stt(b"")] == yielded
    assert reports == [None]


def test_the_notification_text_is_an_argument_not_part_of_the_script() -> None:
    command = notification_command('-say "hi" \\ now')
    assert command[-2:] == ["--", '-say "hi" \\ now']
    assert all('"hi"' not in part for part in command[:-1])


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
    pipeline: status.PipelineState | None, pid: int | str, written_ago: int, crashed: bool, tmp_path: Path, dead_pid: Callable[[], int]
) -> None:
    home = Home(tmp_path)
    now = datetime.now(UTC)
    if pipeline is not None:
        status.write(home.status, status.Status(dead_pid() if pid == GONE else int(pid), now, now - timedelta(seconds=written_ago), status.HEARTBEAT, pipeline, None, 0))
    assert crashed_before(home) is crashed


def test_a_heartbeat_whose_pid_a_later_process_took_is_a_crash(tmp_path: Path) -> None:
    home = Home(tmp_path)
    day_ago = datetime.now(UTC) - timedelta(days=1)
    # This process holds the pid, but it started long after the run that wrote the heartbeat.
    status.write(home.status, status.Status(os.getpid(), day_ago, day_ago, status.HEARTBEAT, "running", None, 0))
    assert crashed_before(home) is True


def test_a_heartbeat_that_does_not_parse_is_not_taken_for_a_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.status.write_text("{")
    assert crashed_before(home) is False
    assert "not counted as a crash" in capsys.readouterr().err
