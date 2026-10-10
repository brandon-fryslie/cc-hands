"""Whisper as LowTalker serves it: each hold uploaded to a fake of LowTalker's POST /v1/audio/transcriptions, which
answers as the network build does (low-talker feed40f), and what hands takes as said from the answer."""

import asyncio
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO
from types import SimpleNamespace
from typing import cast

import pytest
from aiohttp import web
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor

from conftest import by_hand

from hands.sessions.audit import Entry, HoldHeard, Levels, Unsaid
from hands.sessions.wide import WideEvent
from hands.voice import system, transcription, whisper as whisper_module
from hands.voice.microphone import Devices
from hands.voice.pipeline import Voice
from hands.voice.transcription import Busy, Fault, Loading, Lost, NotServing, ServerError, Unanswered
from hands.voice.turnstop import Hold, TurnResolved
from hands.voice.whisper import Whisper, _Recorded  # pyright: ignore[reportPrivateUsage]  (what a release queues)


@dataclass
class Upload:
    """The fields of one upload, as the server read them."""

    fields: dict[str, str]
    filename: str
    content_type: str
    audio: bytes


class Late:
    """An answer the server gives a second after it was asked, long past the hold's patience in these tests."""


@dataclass
class LowTalker:
    """The fake server: what it was sent, and the answers it gives, oldest first: JSON, a body as it is, or one too late."""

    url: str
    answers: list[tuple[int, object]] = field(default_factory=list[tuple[int, object]])
    uploads: list[Upload] = field(default_factory=list[Upload])


def heard(*segments: tuple[str, float, float]) -> object:
    """A verbose_json answer with these segments of text, compression ratio and average log probability."""
    return {
        "task": "transcribe", "language": "english", "duration": 1.5, "text": " ".join(text for text, _, _ in segments),
        "segments": [{"id": index, "start": 0.0, "end": 1.0, "text": text, "compression_ratio": ratio, "avg_logprob": logprob} for index, (text, ratio, logprob) in enumerate(segments)],
    }


# What LowTalker answers a hold with nothing said in it: empty text, no segment (hands-dictation-d7i).
NOTHING: dict[str, object] = {"task": "transcribe", "language": "english", "duration": 1.5, "text": "", "segments": []}


@pytest.fixture
async def lowtalker() -> AsyncIterator[LowTalker]:
    server = LowTalker(url="")

    async def transcriptions(request: web.Request) -> web.Response:
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        server.uploads.append(Upload({name: value for name, value in form.items() if isinstance(value, str)}, upload.filename, upload.content_type, upload.file.read()))
        status, answer = server.answers.pop(0)
        match answer:
            case Late():
                await asyncio.sleep(1.0)
                return web.json_response(NOTHING)
            case bytes() as body:
                return web.Response(body=body, status=status, content_type="text/html")
            case _:
                return web.json_response(answer, status=status)

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcriptions)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    server.url = f"http://127.0.0.1:{runner.addresses[0][1]}/v1"
    yield server
    await runner.cleanup()


def primed(*prompts: str | None) -> Callable[[], Awaitable[str | None]]:
    remaining = iter(prompts)

    async def prompt() -> str | None:
        return next(remaining)

    return prompt


def wav(seconds: float, rate: int = 16_000) -> bytes:
    """A hold as Pipecat hands it to run_stt: mono 16-bit WAV at the pipeline's input rate, 16 kHz unless said."""
    out = BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(b"\x00\x00" * int(rate * seconds))
    return out.getvalue()


# How loud each hold queued here was, as the key's release measured it.
LEVELS = Levels(captured_dbfs=-12.5, heard_dbfs=-40.0)


async def transcribe(whisper: Whisper, hold: int, audio: bytes) -> list[Frame]:
    whisper._transcribing.append(_Recorded(Hold(hold, "held key", 0), LEVELS))  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    return [frame async for frame in whisper.run_stt(audio)]


async def test_each_hold_is_uploaded_as_wav_for_verbose_json_primed_with_the_vocabulary_as_it_is_then(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, NOTHING), (200, NOTHING), (200, NOTHING)]
    whisper = Whisper(url=lowtalker.url, prompt=primed("authMiddleware", None, "sessionStore"), told=by_hand, record=lambda _: None)
    audio = wav(1.0)

    for hold in (1, 2, 3):
        await transcribe(whisper, hold, audio)

    assert [upload.audio for upload in lowtalker.uploads] == [audio] * 3
    assert {upload.content_type for upload in lowtalker.uploads} == {"audio/wav"}
    # An unprimed hold sends no prompt at all, rather than an empty one.
    assert [upload.fields for upload in lowtalker.uploads] == [
        {"model": "whisper-1", "language": "en", "response_format": "verbose_json", "prompt": "authMiddleware"},
        {"model": "whisper-1", "language": "en", "response_format": "verbose_json"},
        {"model": "whisper-1", "language": "en", "response_format": "verbose_json", "prompt": "sessionStore"},
    ]


async def test_what_a_primed_whisper_makes_of_noise_is_not_said(lowtalker: LowTalker) -> None:
    # What LowTalker answered, primed: room noise, Gaussian noise read as a guess and as a loop, and "okay" said quietly
    # at -25 dBFS (hands-dictation-2bs.1xq, hands-dictation-d7i).
    lowtalker.answers += [
        (200, NOTHING),
        (200, heard((".", 0.11, -0.5))),
        (200, heard(("and slow-talking.", 0.68, -2.84))),
        (200, heard(("and turnstop, " * 12, 17.1, -0.24))),
        (200, heard(("Okay.", 0.38, -1.11))),
    ]
    recorded: list[Entry] = []
    whisper = Whisper(url=lowtalker.url, prompt=primed(*["authMiddleware"] * 5), told=by_hand, record=recorded.append)

    said = [frame.text for hold in range(1, 6) for frame in await transcribe(whisper, hold, wav(1.0)) if isinstance(frame, TranscriptionFrame)]

    assert said == ["Okay."]
    # Each hold is recorded with what was dropped from it and why, so a hold that sent nothing can be looked into.
    assert [entry for entry in recorded if isinstance(entry, HoldHeard)] == [
        HoldHeard(1, None, (), LEVELS),
        HoldHeard(2, None, (Unsaid(".", 0.11, -0.5),), LEVELS),
        HoldHeard(3, None, (Unsaid("and slow-talking.", 0.68, -2.84),), LEVELS),
        HoldHeard(4, None, (Unsaid(("and turnstop, " * 12).strip(), 17.1, -0.24),), LEVELS),
        HoldHeard(5, "Okay.", (), LEVELS),
    ]


async def test_a_hold_the_server_refuses_is_said_as_an_error_naming_its_answer_and_whisper_is_done_with_it(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(503, {"error": {"message": "The model is still loading.", "type": "server_error", "param": None, "code": "model_not_ready"}})]
    recorded: list[Entry] = []
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=recorded.append)

    frames = await transcribe(whisper, 7, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame)
    assert "hold 7" in error.error and "503" in error.error and "model_not_ready: The model is still loading." in error.error
    assert not [entry for entry in recorded if isinstance(entry, HoldHeard)]


async def test_an_answer_that_is_not_verbose_json_is_an_error_not_a_hold_heard_as_nothing(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, {"text": "hello"})]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=lambda _: None)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame) and "not verbose_json" in error.error


async def test_a_server_that_is_not_running_is_an_error() -> None:
    # Port 9 (discard) on loopback: nothing listens.
    whisper = Whisper(url="http://127.0.0.1:9/v1", prompt=primed(None), told=by_hand, record=lambda _: None)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]


async def test_an_error_page_that_is_not_json_is_said_with_its_status_and_the_server(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(404, b"<html>404 Not Found</html>")]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=lambda _: None)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame) and f"{lowtalker.url} answered 404" in error.error and "404 Not Found" in error.error


async def test_a_segment_whose_text_is_not_text_is_an_error_not_something_said(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, {"task": "transcribe", "language": "english", "duration": 1.5, "text": "", "segments": [{"text": None, "compression_ratio": 0.38, "avg_logprob": -0.5}]})]
    recorded: list[Entry] = []
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=recorded.append)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame) and "lacking its text or a score" in error.error
    assert not [entry for entry in recorded if isinstance(entry, HoldHeard)]


async def test_a_server_that_never_answers_fails_the_hold_rather_than_holding_up_the_ones_after_it(lowtalker: LowTalker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(whisper_module, "ANSWER_SECONDS", 0.2)
    lowtalker.answers += [(200, Late()), (200, heard(("Okay.", 0.38, -0.5)))]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None, None), told=by_hand, record=lambda _: None)

    stuck = await transcribe(whisper, 1, wav(1.0))
    after = await transcribe(whisper, 2, wav(1.0))

    assert [type(frame) for frame in stuck] == [ErrorFrame, TurnResolved]
    error = stuck[0]
    assert isinstance(error, ErrorFrame) and "did not answer within" in error.error
    assert [frame.text for frame in after if isinstance(frame, TranscriptionFrame)] == ["Okay."]


# What is said when LowTalker cannot transcribe: each way it fails a hold, carried from the upload to the sentence.

NOT_SERVING = "LowTalker is not serving transcription. Start LowTalker's network build and switch Serve Transcription on in its menu"
LOADING = "LowTalker's model is still loading. Speak again once LowTalker's menu says the model is ready"


def said_of(whisper: Whisper, frames: list[Frame]) -> str:
    """What the system channel says of the ErrorFrame a hold ended in, as the pipeline routes Whisper's errors."""
    [error] = [frame for frame in frames if isinstance(frame, ErrorFrame)]
    routed = ErrorFrame(error.error, exception=error.exception, processor=whisper)
    match system.alarm(routed, stt=whisper, llm=FrameProcessor(), tts=FrameProcessor()):
        case system.Say(fact=fact):
            return system.system_text(fact)
        case other:
            raise AssertionError(f"a failed hold is said, not {other}")


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        ((503, {"error": {"message": "model not ready", "code": "model_not_ready"}}), f"That turn was not heard: {LOADING}."),
        ((429, {"error": {"message": "four transcriptions in flight", "code": "rate_limited"}}), "That turn was not heard: LowTalker is already transcribing four things at once. Say it again in a moment."),
        ((200, Late()), "That turn was not heard: LowTalker did not answer within 0.2 seconds. Say it again, and restart LowTalker if it keeps happening."),
        ((500, b"decode failed"), "That turn was not heard: LowTalker failed on its side. Say it again, and restart LowTalker if it keeps happening."),
        (
            (404, b"<html>404 Not Found</html>"),
            "That turn was not heard: LowTalker answered with something that is not a transcription. Check that the transcription url in hands' config.toml is LowTalker's.",
        ),
    ],
)
async def test_each_way_lowtalker_fails_a_hold_is_said_with_what_to_do(lowtalker: LowTalker, monkeypatch: pytest.MonkeyPatch, answer: tuple[int, object], said: str) -> None:
    monkeypatch.setattr(whisper_module, "ANSWER_SECONDS", 0.2)
    lowtalker.answers += [answer]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=lambda _: None)

    assert said_of(whisper, await transcribe(whisper, 1, wav(1.0))) == said


async def test_lowtalker_not_running_is_said_on_the_hold_it_costs() -> None:
    # Nothing listens on the discard port: LowTalker quit, or the offline build, which serves nothing.
    whisper = Whisper(url="http://127.0.0.1:9/v1", prompt=primed(None), told=by_hand, record=lambda _: None)

    assert said_of(whisper, await transcribe(whisper, 1, wav(1.0))) == f"That turn was not heard: {NOT_SERVING}."


async def test_the_start_is_told_the_fault_a_hold_would_meet_now_and_the_probe_is_an_event(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, NOTHING), (503, {"error": {"message": "model not ready", "code": "model_not_ready"}})]
    recorded: list[Entry] = []
    serving = Whisper(url=lowtalker.url, prompt=primed(), told=by_hand, record=recorded.append)
    absent = Whisper(url="http://127.0.0.1:9/v1", prompt=primed(), told=by_hand, record=recorded.append)

    assert await serving.fault() is None
    assert await serving.fault() == Loading("model_not_ready: model not ready")
    assert isinstance(await absent.fault(), NotServing)
    # Each probe is one event, carrying the fault whole: the server's refusal and the connect's reason with it.
    probed = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert [(event.event, event.outcome, event.facts["url"]) for event in probed] == [
        ("transcription.probed", "ok", lowtalker.url),
        ("transcription.probed", "ok", lowtalker.url),
        ("transcription.probed", "ok", "http://127.0.0.1:9/v1"),
    ]
    assert [event.facts["fault"] for event in probed[:2]] == [None, Loading("model_not_ready: model not ready")]
    assert isinstance(probed[2].facts["fault"], NotServing)


@dataclass
class Started:
    """A Voice as far as `system.listen` reads it as the pipeline starts: its handlers, its devices, and its Whisper."""

    fault: Fault | None
    handlers: dict[str, Callable[..., Awaitable[None]]] = field(default_factory=dict[str, Callable[..., Awaitable[None]]])
    said: list[system.SystemFact] = field(default_factory=list[system.SystemFact])

    def event_handler(self, name: str) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
        def register(handler: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
            self.handlers[name] = handler
            return handler

        return register

    async def say(self, fact: system.SystemFact) -> None:
        self.said.append(fact)

    async def start(self) -> list[system.SystemFact]:
        async def fault() -> Fault | None:
            return self.fault

        up = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")
        voice = SimpleNamespace(worker=self, audio=SimpleNamespace(devices=up), stt=SimpleNamespace(fault=fault))
        system.listen(cast("Voice", voice), cast("system.SystemChannel", self), after_crash=False)
        await self.handlers["on_pipeline_started"](self, None)
        return self.said


@pytest.mark.parametrize(
    ("fault", "deaf"),
    [
        (None, ()),
        (NotServing("refused"), (system.Deaf(NotServing("refused")),)),
        (Loading("model not ready"), (system.Deaf(Loading("model not ready")),)),
        # Passing: the next hold may well be heard, and if it is not, that hold says why.
        (Busy("too many"), ()),
        (Unanswered(10.0), ()),
        (Lost("ServerDisconnectedError"), ()),
        (ServerError("500: decode failed"), ()),
    ],
)
async def test_the_start_is_said_first_and_then_only_a_fault_that_fails_every_hold(fault: Fault | None, deaf: tuple[system.SystemFact, ...]) -> None:
    up = Devices(input="MacBook Pro Microphone", output="MacBook Pro Speakers")

    assert await Started(fault).start() == [system.Started(False, up), *deaf]


async def test_a_server_refusing_a_hold_with_a_4xx_leaves_whisper_transcribing_the_next(lowtalker: LowTalker) -> None:
    # A permanent error would make Whisper unusable, and Pipecat then gives it no hold to transcribe or fail aloud.
    lowtalker.answers += [(404, b"<html>404 Not Found</html>")]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=lambda _: None)
    [error] = [frame for frame in await transcribe(whisper, 1, wav(1.0)) if isinstance(frame, ErrorFrame)]

    await whisper.push_error_frame(error)

    assert error.category is not None and not error.category.is_permanent
    assert whisper.is_usable


async def test_whisper_settings_say_the_model_and_language_its_holds_are_uploaded_with() -> None:
    """Pipecat checks a service's settings at its start and logs an error for any it was never given."""
    whisper = Whisper(url="http://127.0.0.1:9/v1", prompt=primed(None), told=by_hand, record=lambda _: None)
    errors: list[str] = []
    sink = logger.add(errors.append, level="ERROR", filter="pipecat")
    try:
        whisper._settings.validate_complete()  # pyright: ignore[reportPrivateUsage]  (what Pipecat's start checks)
    finally:
        logger.remove(sink)

    assert errors == []
    assert whisper._settings.model == transcription.MODEL  # pyright: ignore[reportPrivateUsage]
    assert whisper._settings.language == transcription.LANGUAGE  # pyright: ignore[reportPrivateUsage]


async def test_a_hold_at_a_rate_whisper_does_not_hear_is_said_as_an_error_and_never_uploaded(lowtalker: LowTalker) -> None:
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), told=by_hand, record=lambda _: None)

    failed = await transcribe(whisper, 3, wav(1.0, rate=48_000))

    [error] = [frame for frame in failed if isinstance(frame, ErrorFrame)]
    assert "48000 Hz" in error.error and lowtalker.uploads == []
