"""Whisper as LowTalker serves it: each hold uploaded to a fake of LowTalker's POST /v1/audio/transcriptions, which
answers as the network build does (low-talker feed40f), and what hands takes as said from the answer."""

import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO

import pytest
from aiohttp import web
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame

from hands.sessions.audit import Entry, HoldHeard, Unsaid
from hands.voice.turnstop import TurnResolved
from hands.voice.whisper import Whisper


@dataclass
class Upload:
    """The fields of one upload, as the server read them."""

    fields: dict[str, str]
    filename: str
    content_type: str
    audio: bytes


@dataclass
class LowTalker:
    """The fake server: what it was sent, and the answers it gives, oldest first."""

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


def wav(seconds: float) -> bytes:
    """A hold as Pipecat hands it to run_stt: 16 kHz mono 16-bit WAV."""
    out = BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16_000)
        file.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return out.getvalue()


async def transcribe(whisper: Whisper, hold: int, audio: bytes) -> list[Frame]:
    whisper._transcribing.append(hold)  # pyright: ignore[reportPrivateUsage]  (the hold a release queues)
    return [frame async for frame in whisper.run_stt(audio)]


async def test_each_hold_is_uploaded_as_wav_for_verbose_json_primed_with_the_vocabulary_as_it_is_then(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, NOTHING), (200, NOTHING), (200, NOTHING)]
    whisper = Whisper(url=lowtalker.url, prompt=primed("authMiddleware", None, "sessionStore"), record=lambda _: None)
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
    whisper = Whisper(url=lowtalker.url, prompt=primed(*["authMiddleware"] * 5), record=recorded.append)

    said = [frame.text for hold in range(1, 6) for frame in await transcribe(whisper, hold, wav(1.0)) if isinstance(frame, TranscriptionFrame)]

    assert said == ["Okay."]
    # Each hold is recorded with what was dropped from it and why, so a hold that sent nothing can be looked into.
    assert recorded == [
        HoldHeard(1, None, ()),
        HoldHeard(2, None, (Unsaid(".", 0.11, -0.5),)),
        HoldHeard(3, None, (Unsaid("and slow-talking.", 0.68, -2.84),)),
        HoldHeard(4, None, (Unsaid(("and turnstop, " * 12).strip(), 17.1, -0.24),)),
        HoldHeard(5, "Okay.", ()),
    ]


async def test_a_hold_the_server_refuses_is_said_as_an_error_naming_its_answer_and_whisper_is_done_with_it(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(503, {"error": {"message": "The model is still loading.", "type": "server_error", "param": None, "code": "model_not_ready"}})]
    recorded: list[Entry] = []
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), record=recorded.append)

    frames = await transcribe(whisper, 7, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame)
    assert "hold 7" in error.error and "503" in error.error and "model_not_ready: The model is still loading." in error.error
    assert recorded == []


async def test_an_answer_that_is_not_verbose_json_is_an_error_not_a_hold_heard_as_nothing(lowtalker: LowTalker) -> None:
    lowtalker.answers += [(200, {"text": "hello"})]
    whisper = Whisper(url=lowtalker.url, prompt=primed(None), record=lambda _: None)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
    error = frames[0]
    assert isinstance(error, ErrorFrame) and "not verbose_json" in error.error


async def test_a_server_that_is_not_running_is_an_error() -> None:
    # Port 9 (discard) on loopback: nothing listens.
    whisper = Whisper(url="http://127.0.0.1:9/v1", prompt=primed(None), record=lambda _: None)

    frames = await transcribe(whisper, 1, wav(1.0))

    assert [type(frame) for frame in frames] == [ErrorFrame, TurnResolved]
