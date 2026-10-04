"""`hands run`: the daemon, in the foreground of the terminal it was started in.

    uv run hands run                          # the backend ~/.hands/config.toml names (hands.daemon.config); Sonnet 5 on the Anthropic API when it names none
    uv run --env-file .env hands run          # its key in .env: ANTHROPIC_API_KEY (else the keychain's HANDS_LLM_ANT_KEY) or OPENAI_API_KEY

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
import shlex
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from hands.daemon.config import ANTHROPIC_URL, LLM, Anthropic, Claude, OpenAI, Settings
from hands.sessions import heartbeat
from hands.daemon.notify import post_notification
from hands.sessions.home import Home
from hands.core.front import InFront
from hands.core.place import Modality
from hands.core.wire import UPSTREAM, Answering, Exchanged, Heard, Observed, Sent
from hands.sessions.audit import LLMChosen, ProxyListening, Record, SettingsRead, TapListening, VoiceChosen, failures_to
from hands.sessions.hookconfig import DISPLAY_HOST, DISPLAY_PATH, DISPLAY_PORT, PERMISSION_DEADLINE_SECONDS
from hands.sessions.front import read_front
from hands.sessions.liveness import keep_sweeping, sweep
from hands.sessions.statusfile import keep_reading_statuses
from hands.sessions.tail import Tails, keep_tailing
from hands.sessions.delta import Deltas
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.sessions.proxy import Wire, serve_proxy
from hands.sessions.names import Names
from hands.sessions.server import serve_display, serve_hooks
from hands.sessions.tap import moves, serve_tap
from hands.sessions.overlays import Overlays
from hands.sessions.attention import attention
from hands.voice.devices import follow_default_devices
from hands.voice.cues import RECEIVED, WORKING, QuietCues, cues, keep_cueing
from hands.voice.hold import Move
from hands.voice.keys import drive_quit, drive_talk_key
from hands.voice.phonepage import serve_phone
from hands.voice.floor import Floor
from hands.voice.refocus import Refocus
from hands.voice.vocabulary import Lexicon
from hands.voice.readback import identifier, spoken_name
from hands.voice.pipeline import (
    AnthropicBackend,
    ClaudeCodeBackend,
    LLMBackend,
    OpenAICompatibleBackend,
    Voice,
    VoiceConfig,
    build_llm,
    build_voice,
)
from hands.voice.naming import NAME_INSTRUCTION, NAME_MAX_TOKENS, NAME_TIMEOUT_SECONDS, keep_naming
from hands.voice.narrator import Recounts, attending, narrate
from hands.voice.speech import Pushed, Tailed, Telling, relay
from hands.voice.working import Playing, keep_playing
from hands.voice.summary import Summariser, aside, summariser
from hands.voice.progress_instruction import EXPLAIN_INSTRUCTION, EXPLAIN_MAX_TOKENS, EXPLAIN_TIMEOUT_SECONDS
from hands.voice.sentence_instruction import SENTENCE_INSTRUCTION
from hands.voice.summarising import SENTENCES_MAX_TOKENS, SENTENCES_TIMEOUT_SECONDS, keep_summarising
from hands.voice.sentences import SummaryStore
from hands.voice.briefing import as_sent, brief
from hands.voice.conversation import cue_receipt, record_turns
from hands.voice.system import SystemChannel, listen, told
from hands.threads import off_loop
from hands.daemon.starting import Ended, invocation, keep_beating, start
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION, brain_instruction
from hands.voice.player import Player
from hands.voice import voices
from hands.sessions.payload import Rejected
from hands.voice.ptt import PushToTalk
from hands.voice.tools import Tool, audited, cued, intermediary_tools
from hands.brain.mcp import serve_mcp
from hands.brain.asides import Asides
from hands.brain.process import Brain, Launch, NotLoggedIn, Station, Unstartable, account_kept_out, logged_in, start as start_brain, workdir
from hands.brain.context import EVERY, Keeper, Kept, Store
from hands.brain.stage import BrainStage
from hands.core.session import SessionId

# Where the Anthropic key lives when ANTHROPIC_API_KEY is not set: a generic password in the keychain.
# A prompt to allow access that nobody answers is a failed read, not a daemon that never starts.
KEYCHAIN_TIMEOUT_SECONDS = 30.0
ANTHROPIC_KEYCHAIN_SERVICE = "HANDS_LLM_ANT_KEY"
# How late a permission deadline can be heard.
TICK_SECONDS = 1.0
# How late a session whose process died, or one that started unheard, is noticed.
SWEEP_SECONDS = 2.0
# How late a record Claude Code has written becomes a step of the turn it belongs to. A Stop reads the rest of
# its own transcript before telling the turn, so this is what a turn narrated while it runs waits on, not a Stop.
TAIL_SECONDS = 0.1
# How late Claude Code setting a session's status is heard: the file it rewrites is a few hundred bytes a session.
STATUS_SECONDS = 0.1


def backend(llm: LLM, home: Home, environment: Mapping[str, str]) -> LLMBackend:
    """The backend the settings name, given the key or the login it reaches its model with; raises Rejected naming what it cannot have."""
    # [LAW:single-enforcer] where a setting meets its secret: the key from the environment, or the keychain, or the
    # brain's login, is checked here, once, before the voice loads, rather than once every turn has failed.
    match llm:
        case OpenAI(url=url, model=model):
            return OpenAICompatibleBackend(base_url=url, api_key=_key(environment, "OPENAI_API_KEY"), model=model)
        case Anthropic(url=url, model=model) if url == ANTHROPIC_URL:
            key = _environment_key(environment, "ANTHROPIC_API_KEY") or _keychain_key(ANTHROPIC_KEYCHAIN_SERVICE, "ANTHROPIC_API_KEY")
            return AnthropicBackend(base_url=url, api_key=key, model=model)
        case Anthropic(url=url, model=model):
            # The keychain's key is Anthropic's own, so it is never sent to another server: that server's key is named in the environment.
            return AnthropicBackend(base_url=url, api_key=_key(environment, "ANTHROPIC_API_KEY"), model=model)
        case Claude(model=model):
            try:
                account = logged_in(home.brain, UPSTREAM, environment)
                account_kept_out(home.brain)
            except (NotLoggedIn, Unstartable) as error:
                raise Rejected(str(error)) from error
            return ClaudeCodeBackend(model=model, config_dir=home.brain, account=account)


def _key(environment: Mapping[str, str], var: str) -> str:
    """The API key a keyed backend cannot run without; raises Rejected naming the variable."""
    key = _environment_key(environment, var)
    if not key:
        raise Rejected(f"{var} is not set; the [llm] backend hands is set to run on needs it to reach its model.")
    return key


def _environment_key(environment: Mapping[str, str], var: str) -> str:
    # A key has no whitespace in it: space around one in a .env is dropped, and a blank one is no key.
    return environment.get(var, "").strip()


def _keychain_key(service: str, var: str) -> str:
    """The key the keychain holds under `service`; when it holds none, raises Rejected naming both places a key can be."""
    key = keychain_password(service)
    if not key:
        raise Rejected(f"{var} is not set and the keychain holds no {service}; the [llm] backend hands is set to run on needs one to reach its model.")
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
            raise Rejected(f"reading {service} from the keychain waited {KEYCHAIN_TIMEOUT_SECONDS:.0f}s, likely on a prompt to allow access; {bypass}.")
        finally:
            atexit.unregister(found.kill)
    # [LAW:no-silent-failure] 44 is `security`'s "not found"; any other failure, a locked keychain or a denied prompt, is not an absence.
    if found.returncode == 44:
        return None
    if found.returncode != 0:
        raise Rejected(f"reading {service} from the keychain failed: {err.strip()}; {bypass}.")
    return out.strip() or None


@dataclass(frozen=True)
class Configured:
    """The configuration a run starts with, and the settings it was read from."""

    voice: VoiceConfig
    settings: Settings


def configured_from(home: Home, settings: Settings, environment: Mapping[str, str]) -> Configured:
    """The process boundary: the settings the run started on and the environment's secrets in, typed configuration out."""
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    if stray := sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME"):
        sys.exit(f"hands: {', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}")
    try:
        llm = backend(settings.config.llm, home, environment)
    except Rejected as error:
        sys.exit(f"hands: {error}")
    return Configured(VoiceConfig(llm=llm, whisper_model=settings.config.whisper_model, voice=_voice(home)), settings)


def _voice(home: Home) -> voices.Voice:
    """The voice the user kept; a kept name hands has no voice for, or a file it cannot read, stops the process naming it."""
    try:
        return voices.chosen(home)
    except (Rejected, OSError) as error:
        sys.exit(str(error))


@dataclass(frozen=True)
class Watch:
    """A background task the model's variant brings to the run, which stops the run if it ends."""

    name: str
    run: Callable[[], Coroutine[object, object, None]]


@dataclass(frozen=True)
class Mind:
    """The pipeline's LLM stage for the model's variant, what that variant runs beside the pipeline, and how the model is told how the sessions stand."""

    llm: FrameProcessor
    watches: Sequence[Watch]
    telling: Telling
    # A summariser on this model, given its instruction, how many tokens it may answer in, and how long it has.
    summariser: Callable[[str, int, float], Summariser]


async def front_now(sessions: Sessions, environment: Mapping[str, str]) -> InFront:
    """What is in front on the Mac's screen among the sessions running now."""
    running = {listing.session.membership.id: (listing.session.membership.pid, identifier(listing)) for listing in sessions.live()}
    return await read_front(running, environment)


@asynccontextmanager
async def mind(
    config: VoiceConfig, tools: Sequence[Tool], tail: Callable[[], str], front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], refocus: Refocus, proxy_url: str, wire: Wire, store: Store,
    fritter: Path, log: Path, recall: str, record: Record, environment: Mapping[str, str],
) -> AsyncGenerator[Mind]:
    """The model for the whole conversation: an API service, or the brain's process, the MCP server it reaches hands
    through, the stage that speaks for it from the wire, the keeper of its context, and what answers hands' side questions."""
    # [LAW:single-enforcer] the one place the backend's variant decides the LLM stage.
    match config.llm:
        case AnthropicBackend() | OpenAICompatibleBackend() as backend:
            yield Mind(
                build_llm(backend, instruction=INTERMEDIARY_INSTRUCTION, max_tokens=config.max_reply_tokens),
                (),
                Pushed(),
                lambda instruction, max_tokens, timeout: summariser(backend, instruction, max_tokens, timeout),
            )
        case ClaudeCodeBackend(model=model, config_dir=config_dir):
            server = await serve_mcp(tools, record)
            try:
                station = Station(config_dir, workdir(config_dir), model, proxy_url, environment)
                brain = await start_brain(Launch(station, brain_instruction(log, config_dir, recall), server.config(), SessionId(str(uuid4())), fritter), record)
                try:
                    # [LAW:single-enforcer] everything hands asks in the background is asked here, of a Claude Code of
                    # its own: nothing but the user's turns and their stops is ever typed into the brain.
                    asides = Asides(station, record)
                    stage = BrainStage(brain, tools, tail, refocus, front, modality, record)
                    keeper = Keeper(brain.session, asides.ask, store, EVERY, record)
                    with wire.joined(Kept(stage, keeper, brain, asides)):
                        watches = (Watch("the brain", lambda: outlived(brain)), Watch("the brain's turns", stage.ask_each), Watch("the brain's context", keeper.keep_asking))
                        # A summary is as long as its Claude Code makes it: only its time is the summary's own.
                        yield Mind(stage, watches, Tailed(), lambda instruction, _max_tokens, timeout: aside(asides.ask, instruction, timeout))
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


def _account(backend: LLMBackend) -> str | None:
    """The subscription account the brain runs on, as it was when the run started; None for a keyed variant, whose key is never said."""
    match backend:
        case AnthropicBackend() | OpenAICompatibleBackend():
            return None
        case ClaudeCodeBackend(account=account):
            return account


async def outlived(brain: Brain) -> None:
    # [LAW:no-silent-failure] a brain that ends while hands runs leaves every question unanswered, so it stops the run.
    code = await brain.exited()
    raise RuntimeError(f"the brain exited ({code}) while hands was running")


async def run(
    configure: Callable[[Mapping[str, str]], Configured], survey: Callable[[], None], home: Home, heart: heartbeat.Heart, record: Record, quit_event: asyncio.Event, after_crash: bool,
    environment: Mapping[str, str],
) -> Ended:
    voice: Voice | None = None
    # [LAW:single-enforcer] one owner lets go of all the run took, in reverse, whichever step of taking it raised: a run
    # that raised still lets go of the socket and of every permission hook waiting on it.
    async with AsyncExitStack() as held:
        # [LAW:no-silent-failure] every error hands logs is an audit line too, wherever it was raised.
        failures = logger.add(failures_to(record), level="ERROR", filter="hands")
        held.callback(logger.remove, failures)
        # What each turn changed in the repository it ran in, which no transcript record need name.
        deltas = Deltas(record, environment)
        sessions = Sessions(permission_deadline=PERMISSION_DEADLINE_SECONDS, clock=time.monotonic, record=record, changes=deltas)
        # Before the hooks are served: a turn that finishes while the models load is named once they have.
        names = Names()
        hooks = await serve_hooks(home, sessions, names, record)
        held.push_async_callback(hooks.cleanup)
        wire = Wire(wire_to(record))
        proxy = await serve_proxy(UPSTREAM, wire.observe, wire.route, clock=time.time)
        held.push_async_callback(proxy.close)
        record(ProxyListening(url=proxy.url, upstream=UPSTREAM))
        # [LAW:one-source-of-truth] the working sessions' exchanges reach the same observer as the brain's, so the log and
        # whatever listens hear one wire; and what they say of a turn reaches the registry as it is heard, as a hook does.
        def tapped(observed: Observed) -> None:
            wire.observe(observed)
            for move in moves(observed):
                sessions.hear(move)

        tap = await serve_tap(home.wire, tapped, record, clock=time.time)
        held.callback(tap.close)
        record(TapListening(path=home.wire))
        display = await serve_display(sessions, DISPLAY_HOST, DISPLAY_PORT, DISPLAY_PATH, record)
        held.push_async_callback(display.cleanup)
        store = SummaryStore(Sentences(home.sentences))
        # [LAW:one-source-of-truth] one holder of each session's last turn: the narrator fills it, tell_turn reads it.
        recounts = Recounts()
        # [LAW:one-source-of-truth] one holder of where playback is: the pipeline's taps move it, the playback tools read it.
        player = Player(record)
        # [LAW:single-enforcer] one mover of the focus to a session just told of, for the model's stage and tell_turn alike.
        refocus = Refocus(sessions, home, record)
        # [LAW:one-source-of-truth] one owner of where the user is: the voice's edges move it, set_modality switches it,
        # and the brain's stage reads it.
        key = PushToTalk(record)
        # [LAW:one-source-of-truth] one queue of the cues owed to silence: the tools, the relay, and the turn's receipt owe
        # them, and the run plays them once its speaker is up and quiet.
        quiet_cues = QuietCues()
        tools = [audited(cued(tool, lambda: quiet_cues.owe(WORKING)), record) for tool in intermediary_tools(sessions, store, home, recounts, player, refocus, key.switch)]
        # [LAW:one-source-of-truth] the one environment the run was handed: the settings' secrets, git's, and the brain's alike.
        config = await start(lambda: configured(lambda: configure(environment), survey, home, sessions, record), heart, sessions.live_count, quit_event)
        if config is not None:
            # [LAW:no-ambient-temporal-coupling] the model is up before the voice is built around its stage.
            async with mind(config, tools, lambda: as_sent(sessions, home), lambda: front_now(sessions, environment), lambda: key.modality, refocus, proxy.url, wire, store, home.fritter, home.audit, shlex.join(invocation(home, "recall")), record, environment) as minded:
                # What Whisper is primed with, read as each hold is transcribed.
                lexicon = Lexicon(sessions, home, environment, record)
                floor = Floor(record, minded.telling, lambda id: spoken_name(sessions, id), sessions.live_sessions)
                voice = await start(lambda: off_loop(lambda: build_voice(config, tools, minded.llm, key, player, floor, refocus, lexicon, record), "the voice load"), heart, sessions.live_count, quit_event)
                if voice is not None:
                    sentences = minded.summariser(SENTENCE_INSTRUCTION, SENTENCES_MAX_TOKENS, SENTENCES_TIMEOUT_SECONDS)
                    await converse(voice, home, sessions, heart, quit_event, after_crash, record, deltas, minded, store, sentences, names, recounts, quiet_cues)
    return Ended(None if voice is None else _wall(voice.audio.output().sounded_at), sessions.live_count())


async def configured(configure: Callable[[], Configured], survey: Callable[[], None], home: Home, sessions: Sessions, record: Record) -> VoiceConfig:
    """The configuration, once what hands is missing has been said and the sessions already running are listed."""
    await off_loop(survey, "the readiness check")
    # A restart is back where it was before the models load: every session with a file and a running process is listed.
    await sweep(home, sessions, frozenset())
    read = await off_loop(configure, "the configuration read")
    config = read.voice
    # [LAW:nothing-unseen] which file the settings came from, and the server, model, and collector they chose, is read from the log,
    # not re-derived from a shell.
    read_from = read.settings.path(home)
    record(SettingsRead(path=None if read_from is None else str(read_from), whisper_model=config.whisper_model, collector=read.settings.config.collector))
    record(LLMChosen(backend=type(config.llm).__name__, base_url=_server(config.llm), model=config.llm.model, account=_account(config.llm)))
    record(VoiceChosen(voice=config.voice))
    return config


async def converse(
    voice: Voice,
    home: Home,
    sessions: Sessions,
    heart: heartbeat.Heart,
    quit_event: asyncio.Event,
    after_crash: bool,
    record: Record,
    deltas: Deltas,
    minded: Mind,
    store: SummaryStore,
    sentences: Summariser,
    names: Names,
    recounts: Recounts,
    quiet_cues: QuietCues,
) -> None:
    """Run the pipeline and what feeds it until the run is told to stop; raises what failed if anything did."""
    pipeline = PipelineWatch(voice.worker)
    tails = Tails(sessions)
    channel = SystemChannel(voice.tts, post_notification, record)
    listen(voice, channel, after_crash)
    record_turns(voice.user_turns, voice.assistant_turns, record)
    cue_receipt(voice.user_turns, lambda: quiet_cues.owe(RECEIVED))
    failures: list[BaseException] = []

    def beat() -> None:
        heart.beat(pipeline.state, _wall(voice.audio.output().sounded_at), sessions.live_count(), listening=voice.key.gate.turn_open, deaf=voice.audio.deaf)

    def stop_if_failed(task: asyncio.Task[None]) -> None:
        # [LAW:no-silent-failure] without the ticker nothing is denied at its deadline, without the sweep a dead
        # session stays listed, without the tail no record becomes a step, without the status reader no status Claude Code sets is heard, without the relay
        # nothing is asked aloud, without the progress player the focus is not heard working, without the narrator no finished turn or ended session is heard, without the summary store no backlog or unheard turn is ever said, without the namer no session is given a name, without the heartbeat the daemon looks dead while it runs, without the cue player a turn received and hands acting are never heard, without the device follower an unplugged headset leaves it deaf and mute, and without the talk key no turn starts, so any of
        # them failing stops the run where it can be seen: in its terminal, and as down to the shim and the indicator.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error(f"{task.get_name()} failed; stopping")
            failures.append(error)
            quit_event.set()

    await brief(sessions, home, minded.telling, voice.worker.queue_frame)
    # First sight of every project a session is already working in: its backlog is said before anyone asks for it.
    for listing in sessions.live():
        store.want(listing.session.membership.cwd)
    overlays = Overlays(home)
    playing = Playing()
    background = [
        asyncio.create_task(sessions.keep_time(TICK_SECONDS), name="the permission deadline ticker"),
        asyncio.create_task(keep_sweeping(home, sessions, SWEEP_SECONDS), name="the session liveness sweep"),
        asyncio.create_task(keep_tailing(tails, TAIL_SECONDS, sessions.apply), name="the transcript tail"),
        asyncio.create_task(keep_reading_statuses(sessions.live_ids, sessions.live_session, sessions.now, STATUS_SECONDS, sessions.apply), name="the status reader"),
        asyncio.create_task(relay(sessions, voice.worker.queue_frame, record, partial(attending, home, overlays, lambda: attention(home)), lambda progress, amount: playing.put_nowait((progress, amount)), lambda: quiet_cues.owe(WORKING)), name="the session speech relay"),
        asyncio.create_task(keep_cueing(quiet_cues, voice.audio.output(), record), name="the cues for silence"),
        asyncio.create_task(
            keep_playing(playing, sessions.live_session, voice.worker.queue_frame, record, minded.summariser(EXPLAIN_INSTRUCTION, EXPLAIN_MAX_TOKENS, EXPLAIN_TIMEOUT_SECONDS)),
            name="the progress player",
        ),
        asyncio.create_task(narrate(sessions, tails, voice.worker.queue_frame, record, lambda: attention(home), overlays, recounts, changes=deltas), name="the session narrator"),
        asyncio.create_task(keep_summarising(store, sentences, record), name="the summary store"),
        asyncio.create_task(keep_naming(names, sessions.live_members, minded.summariser(NAME_INSTRUCTION, NAME_MAX_TOKENS, NAME_TIMEOUT_SECONDS), record), name="the namer"),
        asyncio.create_task(keep_beating(beat, heart.period.total_seconds()), name="the heartbeat"),
        *(asyncio.create_task(watch.run(), name=watch.name) for watch in minded.watches),
    ]
    following = asyncio.create_task(follow_default_devices(pipeline.started, voice.audio, channel.say), name="the audio device follower")
    background.append(following)
    for task in background:
        task.add_done_callback(stop_if_failed)

    def after_move(taken: Move) -> None:
        # The move has been made on the gate, as the gate took it, so the tone is the turn's and plays where it is.
        for cue in cues(taken):
            logger.info(cue.line)
            voice.audio.output().cue(cue)
        # The indicator reads the key from the heartbeat, so the edge is written now rather than at the next beat.
        beat()

    async def at_desk(move: Move) -> None:
        match voice.key.move(move, "desk"):
            case None:
                # A Shift at the desk while the turn is the phone's: nothing of it is cued, said, or beaten.
                return
            case taken:
                after_move(taken)
                for fact in told(taken, voice.audio.devices):
                    await channel.say(fact)

    async def drive_talk_key_once_started() -> None:
        # [LAW:no-ambient-temporal-coupling] a move reads the devices, which are known once the pipeline has opened
        # its streams; the key is watched from then on.
        await pipeline.started.wait()
        await drive_talk_key(at_desk)

    async def answer_the_phone_once_started() -> None:
        # As for the talk key: a call is taken once the pipeline is up to hear it.
        await pipeline.started.wait()
        async def cue_the_phone() -> None:
            while True:
                # The phone made the move on the gate as it arrived, in order with its audio.
                after_move(await voice.phone.moves.get())

        try:
            # Either failing ends the other, and the phone task with them.
            async with asyncio.TaskGroup() as phone_tasks:
                phone_tasks.create_task(serve_phone(voice.phone, home, record), name="the phone's page")
                phone_tasks.create_task(cue_the_phone(), name="the phone's cues")
        finally:
            await voice.phone.stop()

    talk_key = asyncio.create_task(drive_talk_key_once_started(), name="the talk key")
    talk_key.add_done_callback(stop_if_failed)
    background.append(talk_key)
    phone = asyncio.create_task(answer_the_phone_once_started(), name="the phone")
    phone.add_done_callback(stop_if_failed)
    background.append(phone)
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
        phone.cancel()
        await asyncio.wait({phone})
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
            case Sent() | Heard() | Answering():
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


def _wall(instant: float | None) -> datetime | None:
    """A monotonic instant as the wall-clock time a reader of the heartbeat can compare with its own."""
    return None if instant is None else datetime.now(UTC) - timedelta(seconds=time.monotonic() - instant)


