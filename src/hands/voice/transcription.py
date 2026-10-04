"""One hold's upload to the transcription server, and the segments it answers with; nothing here loads Pipecat.

The voice transcribes every hold through it, and its start and `hands check` a quarter second of silence (`probe`), so
what they report is of the upload the voice makes and the answer it reads.
"""

import io
import json
import wave
from dataclasses import dataclass
from typing import cast

import aiohttp

from hands.sessions.audit import Unsaid

# What every hold is said in.
LANGUAGE = "en"

# The model the upload names. OpenAI's contract requires one; LowTalker transcribes with the one it holds, whatever is named.
_MODEL = "whisper-1"

# How long the probe waits on its quarter second of silence, which LowTalker with its model loaded answers in
# milliseconds. A probe that runs out has only found the server slow (`Unanswered`, a passing fault), so a bound well
# under a hold's own costs no finding, and it is all the start and `hands check` wait on the server.
PROBE_SECONDS = 10.0


@dataclass(frozen=True)
class NotServing:
    """Nothing listens at the server's address: LowTalker quit, its Serve Transcription is off, or it is the offline build,
    which has no server at all."""

    reason: str


@dataclass(frozen=True)
class Unreachable:
    """The server's address leads nowhere a connection can be made: its host does not resolve, or its TLS does not
    check out. The address is wrong, not LowTalker."""

    reason: str


@dataclass(frozen=True)
class Loading:
    """The server answered 503: LowTalker is up and its model is not loaded yet."""

    refusal: str


@dataclass(frozen=True)
class Busy:
    """The server answered 429: LowTalker transcribes four uploads at once and refuses a fifth."""

    refusal: str


@dataclass(frozen=True)
class Unanswered:
    """The server took the upload and gave no answer within `seconds`."""

    seconds: float


@dataclass(frozen=True)
class Lost:
    """The connection broke after it was made, before an answer came."""

    reason: str


@dataclass(frozen=True)
class Broken:
    """The server answered, and not with a transcription: `answer` is its status and what was wrong with what came with it."""

    answer: str


# [LAW:one-source-of-truth] the one place a fault is judged lasting or not, read by the start and by `hands check` alike.
# A standing fault fails every hold until something is done about it; a passing one may well spare the next hold.
Standing = NotServing | Unreachable | Loading | Broken
Passing = Busy | Unanswered | Lost
Fault = Standing | Passing


class TranscriptionFailed(Exception):
    """[LAW:parse-dont-validate] an upload that came back with no transcription, and why: every way the server can fail
    a hold is named here once, so what is said of it is decided by `fault` and never by reading an error's text."""

    def __init__(self, url: str, fault: Fault) -> None:
        super().__init__(f"{url} {detail(fault)}")
        self.fault = fault


def detail(fault: Fault) -> str:
    """The fault as it is written after the server's address."""
    match fault:
        case NotServing(reason=reason):
            return f"is not listening: {reason}"
        case Unreachable(reason=reason):
            return f"cannot be reached: {reason}"
        case Loading(refusal=refusal):
            return f"answered 503: {refusal}"
        case Busy(refusal=refusal):
            return f"answered 429: {refusal}"
        case Unanswered(seconds=seconds):
            return f"did not answer within {seconds:g} s"
        case Lost(reason=reason):
            return f"dropped the connection: {reason}"
        case Broken(answer=answer):
            return f"answered {answer}"


def remedy(fault: Fault) -> str:
    """What the user can do about the fault, as both the voice and `hands check` say it."""
    match fault:
        case NotServing():
            return "start LowTalker's network build and switch Serve Transcription on in its menu"
        case Unreachable():
            return "check the transcription url in hands' config.toml"
        case Loading():
            return "speak again once LowTalker's menu says the model is ready"
        case Busy():
            return "say it again in a moment"
        case Unanswered():
            return "say it again, and restart LowTalker if it keeps happening"
        case Lost():
            return "say it again"
        case Broken():
            return "check that the transcription url in hands' config.toml is LowTalker's"


async def segments(url: str, wav: bytes, filename: str, prompt: str | None, timeout: float) -> list[Unsaid]:
    """The segments the server at `url` hears in `wav`, primed with `prompt`; raises TranscriptionFailed, naming its
    fault, wherever no transcription came back."""
    form = aiohttp.FormData()
    form.add_field("file", wav, filename=filename, content_type="audio/wav")
    form.add_field("model", _MODEL)
    form.add_field("language", LANGUAGE)
    form.add_field("response_format", "verbose_json")
    # [LAW:dataflow-not-control-flow] no prompt is no field, which is Whisper unprimed.
    for vocabulary in filter(None, (prompt,)):
        form.add_field("prompt", vocabulary)
    # [LAW:no-ambient-temporal-coupling] a session per upload, so nothing has to be opened before the first or closed
    # after the last; on loopback the connection it opens costs nothing a turn would notice.
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session, session.post(f"{url}/audio/transcriptions", data=form) as response:
            status, body = response.status, await response.read()
    # Most specific first: a refused connect is an OSError and a ClientError, and a timeout may be a ClientError too.
    except (aiohttp.ClientConnectorDNSError, aiohttp.ClientSSLError) as error:
        raise TranscriptionFailed(url, Unreachable(f"{type(error).__name__}: {error}")) from error
    except aiohttp.ClientConnectorError as error:
        raise TranscriptionFailed(url, NotServing(str(error.os_error))) from error
    except TimeoutError as error:
        raise TranscriptionFailed(url, Unanswered(timeout)) from error
    except (aiohttp.ClientError, OSError) as error:
        raise TranscriptionFailed(url, Lost(f"{type(error).__name__}: {error}")) from error
    match status:
        case 200:
            return _segments(url, body)
        case 503:
            raise TranscriptionFailed(url, Loading(_refusal(body)))
        case 429:
            raise TranscriptionFailed(url, Busy(_refusal(body)))
        case _:
            raise TranscriptionFailed(url, Broken(f"{status}: {_refusal(body)}"))


async def probe(url: str) -> Fault | None:
    """The fault a hold uploaded now would meet, or None where the server transcribes: a quarter second of silence,
    uploaded as the voice uploads a hold, which LowTalker answers with no segments."""
    silence = io.BytesIO()
    with wave.open(silence, "wb") as written:
        written.setnchannels(1)
        written.setsampwidth(2)
        written.setframerate(16_000)
        written.writeframes(bytes(8_000))
    try:
        await segments(url, silence.getvalue(), "silence.wav", None, PROBE_SECONDS)
    except TranscriptionFailed as failed:
        return failed.fault
    return None


def _segments(url: str, body: bytes) -> list[Unsaid]:
    """[LAW:parse-dont-validate] the segments of a verbose_json answer, each with its text and both scores; raises
    TranscriptionFailed showing the answer where it is not one."""
    try:
        answer: object = json.loads(body)
    except ValueError as error:
        raise TranscriptionFailed(url, Broken(f"200 with an answer that is not JSON: {body[:200]!r}")) from error
    match answer:
        case {"segments": list()}:
            return [_segment(url, segment) for segment in cast("dict[str, list[object]]", answer)["segments"]]
        case _:
            raise TranscriptionFailed(url, Broken(f"200 with an answer that is not verbose_json, having no segments: {body[:200]!r}"))


def _segment(url: str, segment: object) -> Unsaid:
    match segment:
        case {"text": str() as text, "compression_ratio": int() | float() as ratio, "avg_logprob": int() | float() as logprob}:
            return Unsaid(text.strip(), float(ratio), float(logprob))
        case _:
            raise TranscriptionFailed(url, Broken(f"200 with an answer that is not verbose_json, a segment lacking its text or a score: {segment!r:.200}"))


def _refusal(body: bytes) -> str:
    """What an error answer says: an OpenAI-shaped error's code and message, or the body itself where it is not one."""
    try:
        answer: object = json.loads(body)
    except ValueError:
        # Not JSON, as a proxy's or another server's error page is: the body is the refusal.
        return f"{body[:200]!r}"
    match answer:
        case {"error": {"message": str() as message, "code": str() as code}}:
            return f"{code}: {message}"
        case {"error": {"message": str() as message}}:
            return message
        case _:
            return f"{body[:200]!r}"
