"""The heartbeat file: what the daemon last said about itself, and what a reader can conclude from it.

The daemon is the file's only writer; `hands status`, the menu-bar indicator, and the hook shim only read it,
so there is one clock that says whether hands is up. It lives beside the shim, below the daemon, because the shim
reads it and imports only the standard library and hands' data modules.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from hands.sessions.files import replace_whole
from hands.sessions.payload import Payload, Rejected
from hands.sessions.processes import parse_pid, process_starts, still_running

# Starting until Pipecat reports the pipeline started; stopped only in the last heartbeat of a run told to stop.
PipelineState = Literal["starting", "running", "stopped"]


@dataclass(frozen=True)
class Refusal:
    """The last heartbeat of a start hands refused: its pipeline never ran, for `reason`."""

    reason: str

# How often the daemon rewrites the file.
HEARTBEAT = timedelta(seconds=2)
# A reader that has missed this many heartbeats in a row calls the daemon unresponsive.
MISSED_BEATS = 3
# The heartbeat periods a reader believes, in milliseconds: anything past an hour would say nothing about liveness.
PERIODS_MS = range(1, 3_600_001)


@dataclass(frozen=True)
class Degradation:
    """One way an up daemon is running degraded, carrying the words it is said in."""

    # [LAW:one-type-per-behavior] every degradation is only ever said, so each is a value of this type, and the heartbeat
    # carries its words: a new one is a new value, and no reader of the file learns of it.
    brief: str  # a few words for the menu bar, after "hands"
    said: str  # the whole of it, after "hands is up but"


NO_MICROPHONE = Degradation("can't hear", "cannot hear, as there is no microphone")


@dataclass(frozen=True)
class Status:
    pid: int
    started_at: datetime
    written_at: datetime
    heartbeat: timedelta
    pipeline: PipelineState | Refusal
    last_audio_out: datetime | None
    live_sessions: int
    listening: bool  # a turn is open: the talk key is held and has been held long enough to mean talk
    degraded: tuple[Degradation, ...]  # each way it is running degraded, in the order the daemon found them


def encode(status: Status) -> str:
    return json.dumps(
        {
            "pid": status.pid,
            "started_at": status.started_at.isoformat(),
            "written_at": status.written_at.isoformat(),
            "heartbeat_ms": round(status.heartbeat.total_seconds() * 1000),
            **_pipeline_fields(status.pipeline),
            "last_audio_out": None if status.last_audio_out is None else status.last_audio_out.isoformat(),
            "live_sessions": status.live_sessions,
            "listening": status.listening,
            "degraded": [{"brief": each.brief, "said": each.said} for each in status.degraded],
        },
        indent=2,
    )


def parse(raw: bytes) -> Status:
    """The file's contents, parsed once into a Status or refused with what was wrong."""
    # [LAW:parse-dont-validate] every reader of the heartbeat goes through here.
    fields = Payload.parse(raw)
    last_audio_out = fields.optional_text("last_audio_out")
    # A daemon from before degradations said only whether it could not hear, as deaf.
    deaf = (NO_MICROPHONE,) if fields.optional_flag("deaf") else ()
    return Status(
        pid=parse_pid(fields.integer("pid")),
        started_at=_instant(fields.text("started_at")),
        written_at=_instant(fields.text("written_at")),
        heartbeat=_period(fields.integer("heartbeat_ms")),
        pipeline=_pipeline(fields.text("pipeline"), fields.optional_text("refusal")),
        last_audio_out=None if last_audio_out is None else _instant(last_audio_out),
        live_sessions=fields.integer("live_sessions"),
        listening=fields.optional_flag("listening"),
        degraded=(*(_degradation(each) for each in fields.optional_items("degraded")), *deaf),
    )


def write(path: Path, status: Status) -> None:
    """Replace the heartbeat file whole: a reader sees the last heartbeat or this one, never half of either."""
    replace_whole(path, encode(status), 0o600)


@dataclass(frozen=True)
class Heart:
    """What every heartbeat of one run repeats, fixed when the process starts."""

    # [LAW:one-source-of-truth] the pid, start time, and period are decided once, at the process door,
    # so the heartbeat written before Pipecat loads and every one after it agree.
    path: Path
    pid: int
    started_at: datetime
    period: timedelta

    def beat(self, pipeline: PipelineState | Refusal, last_audio_out: datetime | None, live_sessions: int, *, listening: bool, degraded: tuple[Degradation, ...]) -> None:
        write(self.path, Status(self.pid, self.started_at, datetime.now(UTC), self.period, pipeline, last_audio_out, live_sessions, listening, degraded))


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


@dataclass(frozen=True)
class Stopped:
    """The daemon's last heartbeat said its pipeline had finished: it stopped, rather than died or hung."""

    status: Status


@dataclass(frozen=True)
class Refused:
    """The daemon's last heartbeat, written at `written_at`, said it refused to start, and why: it ended before it ran,
    rather than died or hung."""

    refusal: Refusal
    written_at: datetime


@dataclass(frozen=True)
class Unreadable:
    """There is a heartbeat file, but it cannot be read or does not parse: nothing can be said of the daemon."""

    path: Path
    reason: str


Verdict = NeverRan | Up | Unresponsive | Down | Stopped | Refused | Unreadable


def look(path: Path, now: datetime) -> Verdict:
    """What the heartbeat file says of the daemon now: the one way anything outside the daemon learns whether it is up."""
    # [LAW:single-enforcer] `hands status`, the crash check at start, the menu-bar indicator, and the shim all judge through here.
    try:
        last = read(path)
    except (Rejected, OSError) as error:
        # [LAW:no-silent-failure] an unreadable heartbeat is its own verdict, never taken for "not running".
        return Unreadable(path, str(error))
    return judge(path, last, now, alive=last is not None and running(last))


def running(status: Status) -> bool:
    """Whether the process that wrote the heartbeat is still running: its pid is, and not as a later process."""
    # [LAW:single-enforcer] the session sweep's own test for a reused pid. Asked of kill(pid, 0) alone, a pid that
    # went to another process after a crash or a reboot read as a daemon that had stopped responding.
    return still_running(status.pid, status.started_at.timestamp(), process_starts({status.pid}))


def judge(path: Path, status: Status | None, now: datetime, alive: bool) -> Verdict:
    """What the last heartbeat means now, given whether its pid is still a running process."""
    match status:
        case None:
            return NeverRan(path)
        # Checked before the pid: once stopped, a running pid is a daemon still cleaning up, or a stranger reusing the number.
        case Status(pipeline="stopped"):
            return Stopped(status)
        case Status(pipeline=Refusal() as refusal, written_at=written_at):
            return Refused(refusal, written_at)
        case Status() if not alive:
            return Down(status)
        case Status() if now - status.written_at > status.heartbeat * MISSED_BEATS:
            return Unresponsive(status)
        case Status():
            return Up(status)


def describe(verdict: Verdict, now: datetime) -> str:
    match verdict:
        case Unreadable(path=path, reason=reason):
            return f"hands is unknown: its heartbeat at {path} cannot be read: {reason}"
        case NeverRan(path=path):
            return f"hands has not run: there is no heartbeat at {path}"
        case Down(status=status):
            return f"hands is down: its process, pid {status.pid}, is gone; its last heartbeat was {_span(now - status.written_at)} ago"
        case Unresponsive(status=status):
            return (
                f"hands is not responding: pid {status.pid} is running, pipeline {status.pipeline}, "
                f"but its last heartbeat was {_span(now - status.written_at)} ago"
            )
        case Stopped(status=status):
            return f"hands is stopped: pid {status.pid} finished its pipeline {_span(now - status.written_at)} ago"
        case Refused(refusal=Refusal(reason=reason), written_at=written_at):
            return f"hands refused to start {_span(now - written_at)} ago: {reason}"
        case Up(status=status):
            heard = "never" if status.last_audio_out is None else f"{_span(now - status.last_audio_out)} ago"
            state = f"up but {', and '.join(each.said for each in status.degraded)}" if status.degraded else "up"
            return (
                f"hands is {state}: pid {status.pid}, up {_span(now - status.started_at)}, pipeline {status.pipeline}, "
                f"last audio out {heard}, {live_sessions(status)}"
            )


def live_sessions(status: Status) -> str:
    return "1 live session" if status.live_sessions == 1 else f"{status.live_sessions} live sessions"


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


def _period(milliseconds: int) -> timedelta:
    if milliseconds not in PERIODS_MS:
        raise Rejected(f"heartbeat_ms {milliseconds} is not a heartbeat period")
    return timedelta(milliseconds=milliseconds)


def _degradation(item: object) -> Degradation:
    fields = Payload.of(item, "a degradation")
    degradation = Degradation(fields.text("brief"), fields.text("said"))
    if not (degradation.brief and degradation.said):
        raise Rejected(f"a degradation says what is wrong, in brief and in full: {degradation}")
    return degradation


def _pipeline_fields(pipeline: PipelineState | Refusal) -> dict[str, str]:
    match pipeline:
        case Refusal(reason=reason):
            return {"pipeline": "refused", "refusal": reason}
        case state:
            return {"pipeline": state}


def _pipeline(text: str, refusal: str | None) -> PipelineState | Refusal:
    # [LAW:parse-dont-validate] a refusal is read with its reason or not at all: a refused pipeline that names none, or a
    # reason beside a pipeline that ran, is refused here.
    match text, refusal:
        case "starting" | "running" | "stopped", None:
            return text
        case "refused", str():
            return Refusal(refusal)
        case "refused", None:
            raise Rejected("a refused pipeline says why, in refusal")
        case _, str():
            raise Rejected(f"refusal is said only of a refused pipeline, not of {text!r}")
        case other, None:
            raise Rejected(f"pipeline should be starting, running, stopped, or refused, got {other!r}")
