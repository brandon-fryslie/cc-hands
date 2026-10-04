"""One hold's upload to the transcription server, and the segments it answers with; nothing here loads Pipecat.

The voice transcribes every hold through it, and `hands check` a quarter second of silence, so the check is of the
upload the voice makes and the answer it reads.
"""

import json
from typing import cast

import aiohttp

from hands.sessions.audit import Unsaid

# What every hold is said in.
LANGUAGE = "en"

# The model the upload names. OpenAI's contract requires one; LowTalker transcribes with the one it holds, whatever is named.
_MODEL = "whisper-1"


class TranscriptionFailed(Exception):
    """The server answered a hold with something other than its transcription."""


class Refused(TranscriptionFailed):
    """The server answered a hold with an HTTP error."""

    def __init__(self, url: str, status: int, refusal: str) -> None:
        super().__init__(f"{url} answered {status}: {refusal}")
        self.status = status
        self.refusal = refusal


async def segments(url: str, wav: bytes, filename: str, prompt: str | None, timeout: float) -> list[Unsaid]:
    """The segments the server at `url` hears in `wav`, primed with `prompt`; raises TranscriptionFailed where its answer
    is not a transcription, and lets a connection that fails or a timeout raise as itself."""
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
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session, session.post(f"{url}/audio/transcriptions", data=form) as response:
        body = await response.read()
        if response.status != 200:
            raise Refused(url, response.status, _refusal(body))
    return _segments(body)


def _segments(body: bytes) -> list[Unsaid]:
    """[LAW:parse-dont-validate] the segments of a verbose_json answer, each with its text and both scores; raises
    TranscriptionFailed showing the answer where it is not one."""
    try:
        answer: object = json.loads(body)
    except ValueError as error:
        raise TranscriptionFailed(f"the answer is not JSON: {body[:200]!r}") from error
    match answer:
        case {"segments": list()}:
            return [_segment(segment) for segment in cast("dict[str, list[object]]", answer)["segments"]]
        case _:
            raise TranscriptionFailed(f"the answer is not verbose_json, having no segments: {body[:200]!r}")


def _segment(segment: object) -> Unsaid:
    match segment:
        case {"text": str() as text, "compression_ratio": int() | float() as ratio, "avg_logprob": int() | float() as logprob}:
            return Unsaid(text.strip(), float(ratio), float(logprob))
        case _:
            raise TranscriptionFailed(f"the answer is not verbose_json, a segment lacking its text or a score: {segment!r:.200}")


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
