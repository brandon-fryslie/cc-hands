"""`hands run`: the daemon in the foreground, as launchd runs it.

    uv run hands run              # Qwen on inferno (HANDS_LLM=local, the default)
    HANDS_LLM=anthropic ANTHROPIC_API_KEY=... uv run hands run

Sessions join through the hook socket at ~/.hands/hands.sock. A Claude Code
session is registered when its settings carry the hooks this prints:

    uv run python -m hands.sessions.hookconfig

Every heartbeat rewrites ~/.hands/status.json, which `hands status` reads. Run
from a terminal, the space bar holds the conversation: press once to start
talking, press again to stop, `q` quits. Under launchd there is no terminal, so
there is no key edge yet: sessions are registered and spoken about, but not
answered by voice. Latency from key release to the first audio out is logged for
every turn.
"""

import asyncio
import os
import signal
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.whisper.stt import MLXModel
from pipecat.workers.runner import WorkerRunner

from hands.daemon import status
from hands.daemon.notify import post_notification
from hands.sessions.home import Home
from hands.sessions.audit import AuditLog, Record, failures_to
from hands.sessions.hookconfig import PERMISSION_DEADLINE_SECONDS
from hands.sessions.liveness import keep_sweeping, sweep
from hands.sessions.statusfile import keep_reading_statuses
from hands.sessions.tail import Tails, keep_tailing
from hands.sessions.delta import Deltas
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks
from hands.voice.devices import follow_default_devices
from hands.voice.keys import drive_key
from hands.voice.pipeline import (
    AnthropicBackend,
    LLMBackend,
    OpenAICompatibleBackend,
    Voice,
    VoiceConfig,
    build_voice,
)
from hands.voice.ptt import Key
from hands.voice.narrator import narrate
from hands.voice.speech import relay
from hands.voice.summary import Summariser, summariser
from hands.voice.summary_instruction import TURN_SUMMARY_INSTRUCTION
from hands.voice.conversation import record_turns
from hands.voice.system import SystemChannel, listen, unheard
from hands.voice.threads import off_loop
from hands.voice.tools import audited, draft_tools, list_sessions_tool, permission_tools, read_session_tool

# The model lives on inferno, the M4 Max on the LAN, served by mlx_lm.server.
LOCAL_LLM_URL = "http://inferno.local:8080/v1"
LOCAL_LLM_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit"
API_KEY_VAR = "ANTHROPIC_API_KEY"
# How late a permission deadline can be heard.
TICK_SECONDS = 1.0
# How late a session whose process died, or one that started unheard, is noticed.
SWEEP_SECONDS = 2.0
# How late a record Claude Code has written becomes a step of the turn it belongs to. A Stop reads the rest of
# its own transcript before telling the turn, so this is what a turn narrated while it runs waits on, not a Stop.
TAIL_SECONDS = 0.1
# How late Claude Code setting a session's status is heard: the file it rewrites is a few hundred bytes a session.
STATUS_SECONDS = 0.1
# A turn's summary is spoken, so it is short; a model that has not answered in this long is said to have failed.
SUMMARY_MAX_TOKENS = 200
SUMMARY_TIMEOUT_SECONDS = 30.0


def backend_from_env() -> LLMBackend:
    """HANDS_LLM picks the variant: `local` (default) or `anthropic`."""
    # [LAW:parse-dont-validate] the environment is parsed here, once, into a
    # variant that carries exactly what its service needs; an unknown choice
    # or a missing key stops the process at the door.
    choice = os.environ.get("HANDS_LLM", "local")
    if choice == "local":
        return OpenAICompatibleBackend(
            base_url=os.environ.get("HANDS_LLM_URL", LOCAL_LLM_URL),
            model=os.environ.get("HANDS_LLM_MODEL", LOCAL_LLM_MODEL),
        )
    if choice == "anthropic":
        api_key = os.environ.get(API_KEY_VAR)
        if not api_key:
            sys.exit(f"{API_KEY_VAR} is not set; HANDS_LLM=anthropic needs it to reach Claude.")
        return AnthropicBackend(
            api_key=api_key,
            model=os.environ.get("HANDS_LLM_MODEL", "claude-haiku-4-5-20251001"),
        )
    sys.exit(f"HANDS_LLM={choice!r} is not one of: local, anthropic.")


def config_from_env() -> VoiceConfig:
    """The process boundary: environment in, typed configuration out."""
    return VoiceConfig(
        llm=backend_from_env(),
        whisper_model=os.environ.get("HANDS_WHISPER_MODEL", MLXModel.LARGE_V3_TURBO),
        voice=os.environ.get("HANDS_VOICE", "alba"),
    )


async def run(config: VoiceConfig, home: Home, heart: status.Heart, after_crash: bool) -> None:
    audit = AuditLog(home.audit, clock=lambda: datetime.now(UTC))
    # [LAW:no-silent-failure] every error hands logs is an audit line too, wherever it was raised.
    failures = logger.add(failures_to(audit.record), level="ERROR", filter="hands")
    # What each turn changed in the repository it ran in, which no transcript record need name.
    deltas = Deltas()
    sessions = Sessions(permission_deadline=PERMISSION_DEADLINE_SECONDS, clock=time.monotonic, record=audit.record, changes=deltas)
    hooks = await serve_hooks(home, sessions)
    quit_event = asyncio.Event()
    # [LAW:single-enforcer] launchd's SIGTERM, a terminal's Ctrl-C, the q key, and a failed background task all set
    # this one event, and it is installed before the models load, so a stop is heard in every phase of the run.
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, quit_event.set)
    try:
        # A restart is back where it was before the models load: every session with a file and a running process is listed.
        await sweep(home, sessions, frozenset())
        voice = await load(config, sessions, heart, quit_event, audit.record)
        if voice is not None:
            summarise = summariser(config.llm, TURN_SUMMARY_INSTRUCTION, SUMMARY_MAX_TOKENS, SUMMARY_TIMEOUT_SECONDS)
            await converse(voice, home, sessions, summarise, heart, quit_event, after_crash, audit.record, deltas)
    finally:
        # A run that raised still lets go of the socket and of every permission hook waiting on it.
        await hooks.cleanup()
        # From here a signal has its default effect again: nothing is left to stop gracefully.
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signal_number)
        logger.remove(failures)
    # Written only by a stop: a crash leaves the last heartbeat naming a pid that is gone, which reads as down.
    heart.beat("stopped", None if voice is None else _wall(voice.audio.output().sounded_at), sessions.live_count())


async def load(config: VoiceConfig, sessions: Sessions, heart: status.Heart, quit_event: asyncio.Event, record: Record) -> Voice | None:
    """The voice, built off the event loop while the loop beats "starting"; None when told to stop first."""
    tools = [audited(tool, record) for tool in (list_sessions_tool(sessions), read_session_tool(sessions), *draft_tools(sessions), *permission_tools(sessions))]
    # Loading the models takes seconds: off the loop, a slow start reads as starting, and only a stuck loop as not responding.
    building = asyncio.create_task(off_loop(lambda: build_voice(config, tools=tools), "the voice load"))
    starting = asyncio.create_task(keep_beating(lambda: heart.beat("starting", None, sessions.live_count()), heart.period.total_seconds()))
    quitting = asyncio.create_task(quit_event.wait())
    try:
        await asyncio.wait({building, starting, quitting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # A stop does not wait for the models: the load's thread is a daemon, which the process exits without.
        for task in (building, starting, quitting):
            if not task.done():
                task.cancel()
    if starting.done() and not starting.cancelled():
        # [LAW:no-silent-failure] the heartbeat only ends by raising, and its error stops the run as the steady one does.
        starting.result()
    return building.result() if building.done() and not building.cancelled() else None


async def converse(
    voice: Voice,
    home: Home,
    sessions: Sessions,
    summarise: Summariser,
    heart: status.Heart,
    quit_event: asyncio.Event,
    after_crash: bool,
    record: Record,
    deltas: Deltas,
) -> None:
    """Run the pipeline and what feeds it until the run is told to stop; raises what failed if anything did."""
    pipeline = PipelineWatch(voice.worker)
    tails = Tails(sessions)
    channel = SystemChannel(voice.tts, post_notification, record)
    listen(voice, channel, after_crash)
    record_turns(voice.user_turns, voice.assistant_turns, record)
    failures: list[BaseException] = []

    def beat() -> None:
        heart.beat(pipeline.state, _wall(voice.audio.output().sounded_at), sessions.live_count())

    def stop_if_failed(task: asyncio.Task[None]) -> None:
        # [LAW:no-silent-failure] without the ticker nothing is denied at its deadline, without the sweep a dead
        # session stays listed, without the tail no record becomes a step, without the status reader no status Claude Code sets is heard, without the relay
        # nothing is asked aloud, without the narrator no finished turn or ended session is heard, without the heartbeat the daemon looks dead while it runs, without the device follower an unplugged headset leaves it deaf and mute, and without the key edge no turn starts, so any of
        # them failing stops the run where it can be seen, and launchd starts it again.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error(f"{task.get_name()} failed; stopping")
            failures.append(error)
            quit_event.set()

    background = [
        asyncio.create_task(sessions.keep_time(TICK_SECONDS), name="the permission deadline ticker"),
        asyncio.create_task(keep_sweeping(home, sessions, SWEEP_SECONDS), name="the session liveness sweep"),
        asyncio.create_task(keep_tailing(tails, TAIL_SECONDS, sessions.apply), name="the transcript tail"),
        asyncio.create_task(keep_reading_statuses(sessions.live_ids, sessions.live_session, sessions.now, STATUS_SECONDS, sessions.apply), name="the status reader"),
        asyncio.create_task(relay(sessions, voice.worker.queue_frame), name="the session speech relay"),
        asyncio.create_task(narrate(sessions, tails, summarise, voice.worker.queue_frame, record, changes=deltas), name="the session narrator"),
        asyncio.create_task(keep_beating(beat, heart.period.total_seconds()), name="the heartbeat"),
    ]
    following = asyncio.create_task(follow_default_devices(pipeline.started, voice.audio, channel.say), name="the audio device follower")
    background.append(following)
    for task in background:
        task.add_done_callback(stop_if_failed)

    async def on_key(position: Key) -> None:
        turn = voice.key.move_key(position)
        logger.info(f"key {position}: turn {turn}")
        for fact in unheard(turn, voice.audio.devices):
            await channel.say(fact)

    async def drive_key_once_started() -> None:
        # [LAW:no-ambient-temporal-coupling] a press reads the devices, which are known once the pipeline has opened
        # its streams; keys typed before then wait in the terminal.
        await pipeline.started.wait()
        await drive_key(on_key, quit_event)

    if sys.stdin.isatty():
        key_edge = asyncio.create_task(drive_key_once_started(), name="the terminal key edge")
        key_edge.add_done_callback(stop_if_failed)
        background.append(key_edge)
        logger.info("space: press to talk, press again to stop. q: quit.")
    # The run's own signal handler stops the pipeline, so Pipecat installs none of its own.
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    pipeline_run = asyncio.create_task(runner.run(voice.worker))
    quitting = asyncio.create_task(quit_event.wait())
    try:
        await asyncio.wait({pipeline_run, quitting}, return_when=asyncio.FIRST_COMPLETED)
        # [LAW:no-ambient-temporal-coupling] the follower holds the streams while it reopens them, and Pipecat's
        # cleanup closes them; the follower is done before the cleanup starts, so the two never hold them at once.
        following.cancel()
        await asyncio.wait({following})
        await runner.cancel("quit")
        await pipeline_run
    finally:
        quitting.cancel()
        for task in background:
            task.cancel()
    # [LAW:no-silent-failure] a run that failed ends by raising, so it is not written as stopped: it reads as down,
    # exits nonzero, and launchd starts it again.
    if failures:
        raise failures[0]
    if not quit_event.is_set():
        raise RuntimeError("the pipeline ended without being told to stop")


class PipelineWatch:
    """Whether Pipecat has reported the pipeline started, for the heartbeat."""

    def __init__(self, worker: PipelineWorker) -> None:
        # [LAW:single-enforcer] "stopped" is not the watch's to say: a pipeline also finishes while a failed run
        # tears down, and only run() knows the run was told to stop.
        self.state: Literal["starting", "running"] = "starting"
        # Set once, with the state: what waits for the pipeline to have started waits on this.
        self.started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
            self.state = "running"
            self.started.set()


async def keep_beating(beat: Callable[[], None], period: float) -> None:
    """Write the heartbeat now and once a period after, until cancelled."""
    while True:
        beat()
        await asyncio.sleep(period)


def _wall(instant: float | None) -> datetime | None:
    """A monotonic instant as the wall-clock time a reader of the heartbeat can compare with its own."""
    return None if instant is None else datetime.now(UTC) - timedelta(seconds=time.monotonic() - instant)


