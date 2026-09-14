"""The system channel: what hands says about itself, and where it goes when speech or the model is what failed."""

import os
from collections.abc import AsyncGenerator
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
from hands.daemon.notify import notification_command
from hands.sessions.home import Home
from hands.voice.system import (
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
from hands.voice.whisper import NOTHING_TRANSCRIBED, Whisper


@pytest.mark.parametrize(
    ("fact", "said"),
    [
        (Started(after_crash=False), "hands is up."),
        (Started(after_crash=True), "hands is back after a crash."),
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
    assert services.alarm(ErrorFrame("no voice", processor=services.tts)) == Post("hands cannot speak: no voice")
    assert services.alarm(ErrorFrame("device gone", processor=services.transport)) == Unrouted(str(services.transport), "device gone")


class Recorder(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        self.frames: list[Frame] = []

    async def queue_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM, callback: object = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.frames.append(frame)


async def test_a_fact_goes_to_speech_while_it_works_and_to_the_screen_when_it_does_not() -> None:
    tts, posted = Recorder(), list[str]()

    async def notify(text: str) -> None:
        posted.append(text)

    channel = SystemChannel(tts, notify)
    await channel.say(ModelUnreachable())
    assert [(type(frame), getattr(frame, "text", None)) for frame in tts.frames] == [(TTSSpeakFrame, "The language model is unreachable.")]
    await tts.set_usable(False)
    await channel.say(NothingTranscribed())
    await channel.sound(Post("hands cannot speak: no voice"))
    assert len(tts.frames) == 1
    assert posted == ["hands cannot speak, so: Whisper returned nothing for that turn.", "hands cannot speak: no voice"]


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


GONE = 2**22 + 12345


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
    pipeline: status.PipelineState | None, pid: int, written_ago: int, crashed: bool, tmp_path: Path
) -> None:
    home = Home(tmp_path)
    now = datetime.now(UTC)
    if pipeline is not None:
        status.write(home.status, status.Status(pid, now, now - timedelta(seconds=written_ago), status.HEARTBEAT, pipeline, None, 0))
    assert crashed_before(home) is crashed


def test_a_heartbeat_that_does_not_parse_is_not_taken_for_a_crash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = Home(tmp_path)
    home.status.write_text("{")
    assert crashed_before(home) is False
    assert "does not parse" in capsys.readouterr().err
