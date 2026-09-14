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
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.whisper.stt import MLXModel
from pipecat.workers.runner import WorkerRunner

from hands.daemon import status
from hands.sessions.home import Home
from hands.sessions.hookconfig import PERMISSION_DEADLINE_SECONDS
from hands.sessions.registry import Sessions
from hands.sessions.server import serve_hooks
from hands.voice.keys import drive_key
from hands.voice.pipeline import (
    AnthropicBackend,
    LLMBackend,
    OpenAICompatibleBackend,
    VoiceConfig,
    build_voice,
)
from hands.voice.ptt import Key
from hands.voice.speech import relay
from hands.voice.tools import draft_tools, list_sessions_tool, permission_tools

# The model lives on inferno, the M4 Max on the LAN, served by mlx_lm.server.
LOCAL_LLM_URL = "http://inferno.local:8080/v1"
LOCAL_LLM_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit"
API_KEY_VAR = "ANTHROPIC_API_KEY"
# How late a permission deadline can be heard.
TICK_SECONDS = 1.0
# How often status.json is rewritten; a reader calls the daemon unresponsive after a few missed.
HEARTBEAT_SECONDS = 2.0


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


async def run(config: VoiceConfig, home: Home) -> None:
    started_at = datetime.now(UTC)
    sessions = Sessions(permission_deadline=PERMISSION_DEADLINE_SECONDS, clock=time.monotonic)
    hooks = await serve_hooks(home, sessions)
    voice = build_voice(config, tools=[list_sessions_tool(sessions), *draft_tools(sessions), *permission_tools(sessions)])
    pipeline = PipelineWatch(voice.worker)
    quit_event = asyncio.Event()

    def beat() -> None:
        status.write(
            home.status,
            status.Status(
                pid=os.getpid(),
                started_at=started_at,
                written_at=datetime.now(UTC),
                heartbeat=timedelta(seconds=HEARTBEAT_SECONDS),
                pipeline=pipeline.state,
                last_audio_out=_wall(voice.speaker.sounded_at),
                live_sessions=sessions.live_count(),
            ),
        )

    # The first heartbeat goes out before the pipeline loads its models, so a restart shows its new pid at once.
    beat()

    def stop_if_failed(task: asyncio.Task[None]) -> None:
        # [LAW:no-silent-failure] without the ticker nothing is denied at its deadline, without the relay
        # nothing is asked aloud, and without the heartbeat the daemon looks dead while it runs, so any of
        # them failing stops the run where it can be seen, and launchd starts it again.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error(f"{task.get_name()} failed; stopping")
            quit_event.set()

    background = [
        asyncio.create_task(sessions.keep_time(TICK_SECONDS), name="the permission deadline ticker"),
        asyncio.create_task(relay(sessions, voice.worker.queue_frame), name="the session speech relay"),
        asyncio.create_task(keep_beating(beat, HEARTBEAT_SECONDS), name="the heartbeat"),
    ]
    for task in background:
        task.add_done_callback(stop_if_failed)

    async def on_key(position: Key) -> None:
        turn = voice.key.move_key(position)
        logger.info(f"key {position}: turn {turn}")

    runner = WorkerRunner(handle_sigint=True)
    if sys.stdin.isatty():
        background.append(asyncio.create_task(drive_key(on_key, quit_event), name="the terminal key edge"))
        logger.info("space: press to talk, press again to stop. q: quit.")
    pipeline_run = asyncio.create_task(runner.run(voice.worker))
    quitting = asyncio.create_task(quit_event.wait())
    # A signal ends the pipeline without a quit key, and a quit ends the run without a signal: either one stops it.
    await asyncio.wait({pipeline_run, quitting}, return_when=asyncio.FIRST_COMPLETED)
    await runner.cancel("quit")
    await pipeline_run
    quitting.cancel()
    for task in background:
        task.cancel()
    beat()
    await hooks.cleanup()


class PipelineWatch:
    """The pipeline's state as Pipecat reports it, for the heartbeat."""

    def __init__(self, worker: PipelineWorker) -> None:
        self.state: status.PipelineState = "starting"

        @worker.event_handler("on_pipeline_started")
        async def started(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
            self.state = "running"

        @worker.event_handler("on_pipeline_finished")
        async def finished(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
            self.state = "stopped"


async def keep_beating(beat: Callable[[], None], period: float) -> None:
    """Write the heartbeat once a period, until cancelled."""
    while True:
        await asyncio.sleep(period)
        beat()


def _wall(instant: float | None) -> datetime | None:
    """A monotonic instant as the wall-clock time a reader of the heartbeat can compare with its own."""
    return None if instant is None else datetime.now(UTC) - timedelta(seconds=time.monotonic() - instant)


