"""The heartbeat file: what the daemon last said about itself, and what a reader can conclude from it.

The daemon is the file's only writer; `hands status` and the tmux glyph only read it,
so there is one clock that says whether hands is up.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from hands.sessions.payload import Payload, Rejected

# Starting until Pipecat reports the pipeline started, running until it reports it finished.
PipelineState = Literal["starting", "running", "stopped"]

# A reader that has missed this many heartbeats in a row calls the daemon unresponsive.
MISSED_BEATS = 3


@dataclass(frozen=True)
class Status:
    pid: int
    started_at: datetime
    written_at: datetime
    heartbeat: timedelta
    pipeline: PipelineState
    last_audio_out: datetime | None
    live_sessions: int


def encode(status: Status) -> str:
    return json.dumps(
        {
            "pid": status.pid,
            "started_at": status.started_at.isoformat(),
            "written_at": status.written_at.isoformat(),
            "heartbeat_ms": round(status.heartbeat.total_seconds() * 1000),
            "pipeline": status.pipeline,
            "last_audio_out": None if status.last_audio_out is None else status.last_audio_out.isoformat(),
            "live_sessions": status.live_sessions,
        },
        indent=2,
    )


def parse(raw: bytes) -> Status:
    """The file's contents, parsed once into a Status or refused with what was wrong."""
    # [LAW:parse-dont-validate] every reader of the heartbeat goes through here.
    fields = Payload.parse(raw)
    last_audio_out = fields.optional_text("last_audio_out")
    return Status(
        pid=fields.integer("pid"),
        started_at=_instant(fields.text("started_at")),
        written_at=_instant(fields.text("written_at")),
        heartbeat=timedelta(milliseconds=fields.integer("heartbeat_ms")),
        pipeline=_pipeline(fields.text("pipeline")),
        last_audio_out=None if last_audio_out is None else _instant(last_audio_out),
        live_sessions=fields.integer("live_sessions"),
    )


def write(path: Path, status: Status) -> None:
    """Replace the heartbeat file whole: a reader sees the last heartbeat or this one, never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".status-", suffix=".json")
    with os.fdopen(handle, "w") as out:
        out.write(encode(status))
    os.replace(temporary, path)


def read(path: Path) -> Status | None:
    """The last heartbeat, or None when the daemon has never written one."""
    try:
        return parse(path.read_bytes())
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class NeverRan:
    path: Path


@dataclass(frozen=True)
class Up:
    status: Status


@dataclass(frozen=True)
class Unresponsive:
    """The process is alive but has stopped beating: its event loop is stuck."""

    status: Status


@dataclass(frozen=True)
class Down:
    status: Status


Verdict = NeverRan | Up | Unresponsive | Down


def judge(path: Path, status: Status | None, now: datetime, alive: bool) -> Verdict:
    """What the last heartbeat means now, given whether its pid is still a running process."""
    match status:
        case None:
            return NeverRan(path)
        case Status() if not alive:
            return Down(status)
        case Status() if now - status.written_at > status.heartbeat * MISSED_BEATS:
            return Unresponsive(status)
        case Status():
            return Up(status)


def describe(verdict: Verdict, now: datetime) -> str:
    match verdict:
        case NeverRan(path=path):
            return f"hands has not run: there is no heartbeat at {path}"
        case Down(status=status):
            return f"hands is down: pid {status.pid} is not running; its last heartbeat was {_span(now - status.written_at)} ago"
        case Unresponsive(status=status):
            return f"hands is not responding: pid {status.pid} is running, but its last heartbeat was {_span(now - status.written_at)} ago"
        case Up(status=status):
            heard = "never" if status.last_audio_out is None else f"{_span(now - status.last_audio_out)} ago"
            sessions = "1 live session" if status.live_sessions == 1 else f"{status.live_sessions} live sessions"
            return (
                f"hands is up: pid {status.pid}, up {_span(now - status.started_at)}, pipeline {status.pipeline}, "
                f"last audio out {heard}, {sessions}"
            )


def _span(elapsed: timedelta) -> str:
    seconds = max(0, int(elapsed.total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s" if minutes else f"{seconds}s"


def _instant(text: str) -> datetime:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as error:
        raise Rejected(f"not an ISO 8601 time: {text!r}") from error
    if instant.tzinfo is None:
        raise Rejected(f"a heartbeat time carries its zone: {text!r}")
    return instant


def _pipeline(text: str) -> PipelineState:
    match text:
        case "starting" | "running" | "stopped":
            return text
        case other:
            raise Rejected(f"pipeline should be starting, running, or stopped, got {other!r}")
