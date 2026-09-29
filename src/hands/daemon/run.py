"""`hands run`: the daemon, in the foreground of the terminal it was started in.

    uv run hands run              # Sonnet 5 (HANDS_LLM=anthropic, the default); ANTHROPIC_API_KEY, else the keychain's HANDS_LLM_ANT_KEY
    HANDS_LLM=openai uv run --env-file .env hands run    # OPENAI_API_KEY=... in .env
    HANDS_LLM=claude uv run hands run   # a slim Claude Code on the subscription, once: CLAUDE_CONFIG_DIR=~/.hands/brain claude auth login

Sessions join through the hook socket at ~/.hands/hands.sock (the home is
HANDS_HOME when that is set). A Claude Code session is registered when the hands
plugin is installed and enabled; its hooks are plugin/hooks/hooks.json.

Every heartbeat rewrites ~/.hands/status.json, which `hands status` and the
menu-bar indicator read. Right Shift held by itself, in any app, is the talk
key: held for a moment it opens a turn, released it sends it, and any other key
pressed while it is held drops the turn unsent. It needs the Input Monitoring
grant, checked before the run starts. `q` in hands' own terminal quits.
Latency from key release to the first audio out is logged for every turn.
"""

import asyncio
import atexit
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.whisper.stt import MLXModel
from pipecat.workers.runner import WorkerRunner

from hands.sessions import heartbeat
from hands.daemon.notify import post_notification
from hands.sessions.home import Home
from hands.core.wire import Exchanged, Heard, Observed, Sent
from hands.sessions.audit import AuditLog, LLMChosen, ProxyListening, Record, failures_to
from hands.sessions.hookconfig import PERMISSION_DEADLINE_SECONDS
from hands.sessions.liveness import keep_sweeping, sweep
from hands.sessions.statusfile import keep_reading_statuses
from hands.sessions.tail import Tails, keep_tailing
from hands.sessions.delta import Deltas
from hands.sessions.registry import Sessions
from hands.sessions.proxy import UPSTREAM, serve_proxy
from hands.sessions.server import serve_hooks
from hands.sessions.summaries import summaries
from hands.voice.devices import follow_default_devices
from hands.voice.cues import cues
from hands.voice.hold import Move
from hands.voice.keys import drive_quit, drive_talk_key
from hands.voice.pipeline import (
    AnthropicBackend,
    ClaudeCodeBackend,
    LLMBackend,
    OpenAICompatibleBackend,
    Voice,
    VoiceConfig,
    build_voice,
)
from hands.voice.narrator import narrate
from hands.voice.speech import relay
from hands.voice.summary import Summariser, summariser
from hands.voice.summary_instruction import TURN_SUMMARY_INSTRUCTION
from hands.voice.briefing import brief
from hands.voice.conversation import record_turns
from hands.voice.system import SystemChannel, listen, told
from hands.voice.threads import off_loop
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION
from hands.voice.tools import Tool, audited, intermediary_tools
from hands.brain.mcp import serve_mcp
from hands.brain.process import Brain, Launch, NotLoggedIn, logged_in, start as start_brain, workdir

# The SDK appends /v1/messages to this, so an Anthropic-compatible server's URL has no /v1 of its own.
ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_MODEL = "claude-sonnet-5"
# Where the Anthropic key lives when ANTHROPIC_API_KEY is not set: a generic password in the keychain.
# A prompt to allow access that nobody answers is a failed read, not a daemon that never starts.
KEYCHAIN_TIMEOUT_SECONDS = 30.0
ANTHROPIC_KEYCHAIN_SERVICE = "HANDS_LLM_ANT_KEY"
OPENAI_URL = "https://api.openai.com/v1"
# Not a reasoning model, so no thinking precedes the first spoken word; it calls tools and takes max_tokens.
OPENAI_MODEL = "gpt-4.1-mini"
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


def backend_from_env(home: Home) -> LLMBackend:
    """HANDS_LLM picks the variant: `anthropic` (default), `openai`, or `claude`; HANDS_LLM_MODEL moves any, HANDS_LLM_URL the first two."""
    # [LAW:parse-dont-validate] the environment is parsed here, once, into a
    # variant that carries exactly what its service needs; an unknown choice
    # or a missing key stops the process at the door.
    choice = os.environ.get("HANDS_LLM", "anthropic")
    if choice == "openai":
        return OpenAICompatibleBackend(
            base_url=_environment_url() or OPENAI_URL,
            api_key=_key("OPENAI_API_KEY", choice),
            model=os.environ.get("HANDS_LLM_MODEL", OPENAI_MODEL),
        )
    if choice == "anthropic":
        model = os.environ.get("HANDS_LLM_MODEL", ANTHROPIC_MODEL)
        url = _environment_url()
        if url is None:
            key = _environment_key("ANTHROPIC_API_KEY") or _keychain_key(ANTHROPIC_KEYCHAIN_SERVICE, "ANTHROPIC_API_KEY", choice)
            return AnthropicBackend(base_url=ANTHROPIC_URL, api_key=key, model=model)
        # The keychain's key is Anthropic's own, so it is never sent to another server: that server's key is named in the environment.
        if url.rstrip("/").endswith("/v1"):
            sys.exit(f"HANDS_LLM_URL={url!r} ends in /v1, and the Anthropic client appends /v1/messages itself; drop the /v1.")
        return AnthropicBackend(base_url=url, api_key=_key("ANTHROPIC_API_KEY", choice), model=model)
    if choice == "claude":
        # [LAW:no-silent-failure] the brain reaches the API through hands' proxy, which forwards to Anthropic's; a URL
        # named for it would be ignored, so it is refused instead.
        if _environment_url() is not None:
            sys.exit("HANDS_LLM_URL does not apply to HANDS_LLM=claude, whose requests go through hands' proxy to Anthropic's API; unset it.")
        # A brain with no login is refused here, before the voice loads, rather than once every turn has failed.
        try:
            logged_in(home.brain, UPSTREAM)
        except NotLoggedIn as error:
            sys.exit(f"hands: {error}")
        return ClaudeCodeBackend(model=os.environ.get("HANDS_LLM_MODEL", ANTHROPIC_MODEL), config_dir=home.brain)
    sys.exit(f"HANDS_LLM={choice!r} is not one of: anthropic, openai, claude.")


def _key(var: str, choice: str) -> str:
    """The API key a keyed variant cannot run without, or the process stops naming the variable."""
    key = _environment_key(var)
    if not key:
        sys.exit(f"{var} is not set; HANDS_LLM={choice} needs it to reach its model.")
    return key


def _environment_url() -> str | None:
    """The server HANDS_LLM_URL names, or None when it names none: a blank line in a .env is no URL, as a blank key is no key."""
    return os.environ.get("HANDS_LLM_URL", "").strip() or None


def _environment_key(var: str) -> str:
    # A key has no whitespace in it: space around one in a .env is dropped, and a blank one is no key.
    return os.environ.get(var, "").strip()


def _keychain_key(service: str, var: str, choice: str) -> str:
    """The key the keychain holds under `service`; when it holds none, the process stops naming both places a key can be."""
    key = keychain_password(service)
    if not key:
        sys.exit(f"{var} is not set and the keychain holds no {service}; HANDS_LLM={choice} needs one to reach its model.")
    return key


def keychain_password(service: str) -> str | None:
    """The generic password the keychains on the search list hold for `service`, or None when they hold none."""
    bypass = "set ANTHROPIC_API_KEY to start without the keychain"
    with subprocess.Popen(["security", "find-generic-password", "-s", service, "-w"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as found:
        # The read runs on a daemon thread, which a stop mid-prompt exits without: the prompt goes with the process
        # that asked, not left on screen for a daemon that is gone.
        atexit.register(found.kill)
        try:
            out, err = found.communicate(timeout=KEYCHAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            found.kill()
            sys.exit(f"reading {service} from the keychain waited {KEYCHAIN_TIMEOUT_SECONDS:.0f}s, likely on a prompt to allow access; {bypass}.")
        finally:
            atexit.unregister(found.kill)
    # [LAW:no-silent-failure] 44 is `security`'s "not found"; any other failure, a locked keychain or a denied prompt, is not an absence.
    if found.returncode == 44:
        return None
    if found.returncode != 0:
        sys.exit(f"reading {service} from the keychain failed: {err.strip()}; {bypass}.")
    return out.strip() or None


def config_from_env(home: Home) -> VoiceConfig:
    """The process boundary: environment in, typed configuration out."""
    return VoiceConfig(
        llm=backend_from_env(home),
        whisper_model=os.environ.get("HANDS_WHISPER_MODEL", MLXModel.LARGE_V3_TURBO),
        voice=os.environ.get("HANDS_VOICE", "alba"),
    )


@dataclass(frozen=True)
class Watch:
    """A background task the model's variant brings to the run, which stops the run if it ends."""

    name: str
    run: Callable[[], Coroutine[object, object, None]]


@asynccontextmanager
async def mind(backend: LLMBackend, tools: Sequence[Tool], proxy_url: str, record: Record) -> AsyncGenerator[Sequence[Watch]]:
    """What the model's variant runs beside the pipeline for the whole conversation: for the brain, its process and the MCP server it reaches hands through."""
    match backend:
        case AnthropicBackend() | OpenAICompatibleBackend():
            yield ()
        case ClaudeCodeBackend(model=model, config_dir=config_dir):
            server = await serve_mcp(tools, record)
            try:
                launch = Launch(config_dir, workdir(config_dir), model, INTERMEDIARY_INSTRUCTION, proxy_url, server.config())
                brain = await start_brain(launch, record)
                try:
                    yield (Watch("the brain", lambda: outlived(brain)),)
                finally:
                    await brain.stop()
            finally:
                await server.close()


def _server(backend: LLMBackend) -> str:
    """The server a backend's model answers on: the brain's is Anthropic's, reached through hands' proxy."""
    match backend:
        case AnthropicBackend(base_url=base_url) | OpenAICompatibleBackend(base_url=base_url):
            return base_url
        case ClaudeCodeBackend():
            return UPSTREAM


async def outlived(brain: Brain) -> None:
    # [LAW:no-silent-failure] a brain that ends while hands runs leaves every question unanswered, so it stops the run.
    code = await brain.exited()
    raise RuntimeError(f"the brain exited ({code}) while hands was running")


# The signals that stop a run as the q key does: closing its terminal is how a run in a terminal is most often ended.
QUIT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


async def run(configure: Callable[[], VoiceConfig], survey: Callable[[], None], home: Home, heart: heartbeat.Heart, after_crash: bool) -> None:
    audit = AuditLog(home.audit, clock=lambda: datetime.now(UTC))
    # [LAW:no-silent-failure] every error hands logs is an audit line too, wherever it was raised.
    failures = logger.add(failures_to(audit.record), level="ERROR", filter="hands")
    # What each turn changed in the repository it ran in, which no transcript record need name.
    deltas = Deltas()
    sessions = Sessions(permission_deadline=PERMISSION_DEADLINE_SECONDS, clock=time.monotonic, record=audit.record, changes=deltas)
    hooks = await serve_hooks(home, sessions)
    proxy = await serve_proxy(UPSTREAM, wire_to(audit.record), clock=time.time)
    audit.record(ProxyListening(url=proxy.url, upstream=UPSTREAM))
    quit_event = asyncio.Event()
    # [LAW:single-enforcer] a SIGTERM, a terminal's Ctrl-C, the terminal closing (SIGHUP), the q key, and a failed
    # background task all set this one event, and it is installed before the start, so a stop is heard in every phase of the run.
    loop = asyncio.get_running_loop()
    for signal_number in QUIT_SIGNALS:
        loop.add_signal_handler(signal_number, quit_event.set)
    voice: Voice | None = None
    tools = [audited(tool, audit.record) for tool in intermediary_tools(sessions)]
    try:
        started = await start(configure, survey, home, sessions, tools, heart, quit_event, audit.record)
        if started is not None:
            config, voice = started
            async with mind(config.llm, tools, proxy.url, audit.record) as watches:
                summarise = summariser(config.llm, proxy.url, TURN_SUMMARY_INSTRUCTION, SUMMARY_MAX_TOKENS, SUMMARY_TIMEOUT_SECONDS)
                await converse(voice, home, sessions, summarise, heart, quit_event, after_crash, audit.record, deltas, watches)
    finally:
        # A run that raised still lets go of the socket and of every permission hook waiting on it.
        await hooks.cleanup()
        await proxy.close()
        # From here a signal has its default effect again: nothing is left to stop gracefully.
        for signal_number in QUIT_SIGNALS:
            loop.remove_signal_handler(signal_number)
        logger.remove(failures)
    # Written only by a stop: a crash leaves the last heartbeat naming a pid that is gone, which reads as down.
    heart.beat("stopped", None if voice is None else _wall(voice.audio.output().sounded_at), sessions.live_count(), False)


async def start(
    configure: Callable[[], VoiceConfig],
    survey: Callable[[], None],
    home: Home,
    sessions: Sessions,
    tools: Sequence[Tool],
    heart: heartbeat.Heart,
    quit_event: asyncio.Event,
    record: Record,
) -> tuple[VoiceConfig, Voice] | None:
    """The configuration and the voice, while the loop beats "starting"; None when told to stop first.

    [LAW:single-enforcer] one beater says "starting" for the whole start. Its slow steps run off the loop: saying what
    hands is missing asks `claude`, reading the configuration can wait on the user at a keychain prompt, and loading
    the models takes seconds. A start that waits reads as starting, and only a stuck loop as not responding.
    """

    async def prepare() -> tuple[VoiceConfig, Voice]:
        await off_loop(survey, "the readiness check")
        # A restart is back where it was before the models load: every session with a file and a running process is listed.
        await sweep(home, sessions, frozenset())
        config = await off_loop(configure, "the configuration read")
        # [LAW:nothing-unseen] which server and model the environment chose is read from the log, not re-derived from a shell.
        record(LLMChosen(backend=type(config.llm).__name__, base_url=_server(config.llm), model=config.llm.model))
        return config, await off_loop(lambda: build_voice(config, tools=tools), "the voice load")

    preparing = asyncio.create_task(prepare())
    starting = asyncio.create_task(keep_beating(lambda: heart.beat("starting", None, sessions.live_count(), False), heart.period.total_seconds()))
    quitting = asyncio.create_task(quit_event.wait())
    try:
        await asyncio.wait({preparing, starting, quitting}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # A stop does not wait for the models or the keychain: their threads are daemons, which the process exits without.
        for task in (preparing, starting, quitting):
            if not task.done():
                task.cancel()
    if starting.done() and not starting.cancelled():
        # [LAW:no-silent-failure] the heartbeat only ends by raising, and its error stops the run as the steady one does.
        starting.result()
    return preparing.result() if preparing.done() and not preparing.cancelled() else None


async def converse(
    voice: Voice,
    home: Home,
    sessions: Sessions,
    summarise: Summariser,
    heart: heartbeat.Heart,
    quit_event: asyncio.Event,
    after_crash: bool,
    record: Record,
    deltas: Deltas,
    watches: Sequence[Watch],
) -> None:
    """Run the pipeline and what feeds it until the run is told to stop; raises what failed if anything did."""
    pipeline = PipelineWatch(voice.worker)
    tails = Tails(sessions)
    channel = SystemChannel(voice.tts, post_notification, record)
    listen(voice, channel, after_crash)
    record_turns(voice.user_turns, voice.assistant_turns, record)
    failures: list[BaseException] = []

    def beat() -> None:
        heart.beat(pipeline.state, _wall(voice.audio.output().sounded_at), sessions.live_count(), voice.key.gate.turn_open)

    def stop_if_failed(task: asyncio.Task[None]) -> None:
        # [LAW:no-silent-failure] without the ticker nothing is denied at its deadline, without the sweep a dead
        # session stays listed, without the tail no record becomes a step, without the status reader no status Claude Code sets is heard, without the relay
        # nothing is asked aloud, without the narrator no finished turn or ended session is heard, without the heartbeat the daemon looks dead while it runs, without the device follower an unplugged headset leaves it deaf and mute, and without the talk key no turn starts, so any of
        # them failing stops the run where it can be seen: in its terminal, and as down to the shim and the indicator.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error(f"{task.get_name()} failed; stopping")
            failures.append(error)
            quit_event.set()

    await brief(sessions, voice.worker.queue_frame)
    background = [
        asyncio.create_task(sessions.keep_time(TICK_SECONDS), name="the permission deadline ticker"),
        asyncio.create_task(keep_sweeping(home, sessions, SWEEP_SECONDS), name="the session liveness sweep"),
        asyncio.create_task(keep_tailing(tails, TAIL_SECONDS, sessions.apply), name="the transcript tail"),
        asyncio.create_task(keep_reading_statuses(sessions.live_ids, sessions.live_session, sessions.now, STATUS_SECONDS, sessions.apply), name="the status reader"),
        asyncio.create_task(relay(sessions, voice.worker.queue_frame), name="the session speech relay"),
        asyncio.create_task(narrate(sessions, tails, summarise, voice.worker.queue_frame, record, lambda: summaries(home), channel.say, changes=deltas), name="the session narrator"),
        asyncio.create_task(keep_beating(beat, heart.period.total_seconds()), name="the heartbeat"),
        *(asyncio.create_task(watch.run(), name=watch.name) for watch in watches),
    ]
    following = asyncio.create_task(follow_default_devices(pipeline.started, voice.audio, channel.say), name="the audio device follower")
    background.append(following)
    for task in background:
        task.add_done_callback(stop_if_failed)

    async def on_move(move: Move) -> None:
        voice.key.move(move)
        for cue in cues(move):
            logger.info(cue.line)
            voice.audio.output().cue(cue)
        # The indicator reads the key from the heartbeat, so the edge is written now rather than at the next beat.
        beat()
        for fact in told(move, voice.audio.devices):
            await channel.say(fact)

    async def drive_talk_key_once_started() -> None:
        # [LAW:no-ambient-temporal-coupling] a move reads the devices, which are known once the pipeline has opened
        # its streams; the key is watched from then on.
        await pipeline.started.wait()
        await drive_talk_key(on_move)

    talk_key = asyncio.create_task(drive_talk_key_once_started(), name="the talk key")
    talk_key.add_done_callback(stop_if_failed)
    background.append(talk_key)
    logger.info("hold Right Shift to talk, release to send; a key pressed while it is held drops the turn.")
    if sys.stdin.isatty():
        quit_key = asyncio.create_task(drive_quit(quit_event), name="the terminal quit key")
        quit_key.add_done_callback(stop_if_failed)
        background.append(quit_key)
        logger.info("q: quit.")
    # The run's own signal handler stops the pipeline, so Pipecat installs none of its own.
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    pipeline_run = asyncio.create_task(runner.run(voice.worker))
    quitting = asyncio.create_task(quit_event.wait())
    try:
        await asyncio.wait({pipeline_run, quitting}, return_when=asyncio.FIRST_COMPLETED)
        # A run told to stop takes no more turns: the key stops being watched before the pipeline tears down, not after.
        talk_key.cancel()
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
    # exits nonzero, and every hook after it fails loudly until hands is run again.
    if failures:
        raise failures[0]
    if not quit_event.is_set():
        raise RuntimeError("the pipeline ended without being told to stop")


def wire_to(record: Record) -> Callable[[Observed], None]:
    """What the daemon keeps of the wire: one audit line per exchange, the wide record of the proxy's work."""

    def observe(observed: Observed) -> None:
        match observed:
            case Exchanged():
                record(observed)
            case Sent() | Heard():
                # Heard as it happens by what speaks from the wire; the exchange's line already holds the request's kind and the whole reply.
                pass

    return observe


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


