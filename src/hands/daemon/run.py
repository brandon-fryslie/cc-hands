"""`hands run`: the daemon, in the foreground of the terminal it was started in.

    uv run hands run                          # on the brain, logged in with `hands login` (hands.daemon.config)

Sessions join through the hook socket at ~/.hands/hands.sock (the home is
HANDS_HOME when that is set). A Claude Code session is registered when the hands
plugin is installed and enabled; its hooks are src/hands/sessions/plugin/hooks/hooks.json.

Every heartbeat rewrites ~/.hands/status.json, which `hands status` and the
menu-bar indicator read. Right Shift held by itself, in any app, is the talk
key: held for a moment it opens a turn, released it sends it, and any other key
pressed while it is held drops the turn unsent. It needs the Input Monitoring
grant, checked before the run starts. `q` in hands' own terminal quits.
Latency from key release to the first audio out is logged for every turn.
"""

import asyncio
import shlex
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

from hands.daemon.config import OwnModel, Settings
from hands.sessions import heartbeat, tmux
from hands.sessions.typing import type_into
from hands.daemon.notify import post_notification
from hands.sessions.home import Home
from hands.core.front import InFront
from hands.core.place import Modality
from hands.core.wire import UPSTREAM, Answering, Exchanged, Heard, Observed, Sent
from hands.sessions.audit import Record, failures_to
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
from hands.voice.cues import RECEIVED, WORKING, QuietCues, cues
from hands.voice.hold import Move
from hands.voice import wake
from hands.voice.engaged import drive, drive_engaged, loaded, untapped
from hands.voice.wake import WAKE
from hands.voice.wakeword import Trained, Word
from hands.voice.keys import drive_quit, drive_talk_key, tapped
from hands.voice.phonepage import serve_phone
from hands.voice.turnstop import Typed
from hands.voice.floor import Floor
from hands.voice.refocus import Refocus
from hands.voice.vocabulary import Lexicon
from hands.voice.readback import identifier, spoken_name
from hands.voice.pipeline import Voice, VoiceConfig, build_voice
from hands.daemon.backend import backend
from hands.voice.naming import NAME_INSTRUCTION, NAME_TIMEOUT_SECONDS, keep_naming
from hands.voice.narrator import Recounts, attending, narrate
from hands.voice.utterance import Utterances
from hands.voice.speakers import Room, teller
from hands.voice.speech import relay
from hands.voice.working import Playing, keep_playing
from hands.voice.summary import Summariser, aside
from hands.voice.progress_instruction import EXPLAIN_INSTRUCTION, EXPLAIN_TIMEOUT_SECONDS
from hands.voice.sentence_instruction import SENTENCE_INSTRUCTION
from hands.voice.summarising import SENTENCES_TIMEOUT_SECONDS, keep_summarising
from hands.voice.sentences import SummaryStore
from hands.voice.briefing import as_sent
from hands.voice.conversation import cue_receipt, record_turns
from hands.voice.system import SystemChannel, listen, told
from hands.threads import off_loop
from hands.daemon.starting import CannotStart, Ended, Start, invocation, keep_beating, start
from hands.voice.intermediary_instruction import brain_instruction
from hands.voice.player import Player
from hands.voice import voices
from hands.sessions.payload import Rejected
from hands.voice.ptt import PushToTalk
from hands.voice.trigger import Edge, Trigger, Triggers
from hands.voice.tool import Tool
from hands.voice.tools import audited, close_session_tool, cued, handed_back, intermediary_tools, start_session_tool, usage_tool
from hands.brain.mcp import CallSpans, serve_mcp
from hands.brain.asides import AsideKind, Asides
from hands.brain.process import Brain, Launch, Station, Unstartable, conversation, start as start_brain, workdir
from hands.brain.context import EVERY, LINE_TIME, Keeper, Kept, Store
from hands.brain.stage import BrainStage
from hands.brain.usage import Usage
from hands.spotify import Catalogue, credentials
from hands.core.session import Drive, SessionId

# How late a permission deadline can be heard.
TICK_SECONDS = 1.0
# How late a session whose process died, or one that started unheard, is noticed.
SWEEP_SECONDS = 2.0
# How late a record Claude Code has written becomes a step of the turn it belongs to. A Stop reads the rest of
# its own transcript before telling the turn, so this is what a turn narrated while it runs waits on, not a Stop.
TAIL_SECONDS = 0.1
# How late Claude Code setting a session's status is heard: the file it rewrites is a few hundred bytes a session.
STATUS_SECONDS = 0.1


@dataclass(frozen=True)
class Configured:
    """The configuration a run starts with, and the settings it was read from."""

    voice: VoiceConfig
    settings: Settings


def configured_from(home: Home, settings: Settings, environment: Mapping[str, str]) -> Configured:
    """The process boundary: the settings the run started on and the environment's secrets in, typed configuration out;
    CannotStart where hands cannot run on them."""
    try:
        llm = backend(settings.config.llm, home, environment)
    except Rejected as error:
        raise CannotStart(str(error)) from error
    return Configured(VoiceConfig(llm=llm, transcription=settings.config.transcription, voice=_voice(home), personality=settings.config.personality, wake=settings.config.wake), settings)


def _attempted(configure: Callable[[], Configured]) -> Configured | CannotStart:
    """The configuration, or why hands cannot start on the settings, as a value the readiness check can say."""
    try:
        return configure()
    except CannotStart as error:
        return error


def _voice(home: Home) -> voices.Voice:
    """The voice the user kept; a kept name hands has no voice for, or a file it cannot read, stops the process naming it."""
    try:
        return voices.chosen(home)
    except (Rejected, OSError) as error:
        raise CannotStart(str(error)) from error


@dataclass(frozen=True)
class Watch:
    """A background task the model's variant brings to the run, which stops the run if it ends."""

    name: str
    run: Callable[[], Coroutine[object, object, None]]


@dataclass(frozen=True)
class Mind:
    """The pipeline's LLM stage, what the brain runs beside the pipeline, and what answers hands' side questions."""

    llm: FrameProcessor
    watches: Sequence[Watch]
    # A summariser, given what it is for, its instruction, and how long it has.
    summariser: Callable[[AsideKind, str, float], Summariser]


async def front_now(sessions: Sessions, environment: Mapping[str, str]) -> InFront:
    """What is in front on the Mac's screen among the sessions running now."""
    running = {listing.session.membership.id: (listing.session.membership.pid, identifier(listing)) for listing in sessions.live()}
    return await read_front(running, environment)


@asynccontextmanager
async def mind(
    config: VoiceConfig, tools: Sequence[Tool], brain_tools: Sequence[Tool], tail: Callable[[], str], front: Callable[[], Awaitable[InFront]], modality: Callable[[], Modality], opened: Callable[[], Edge], refocus: Refocus, handed: Callable[[SessionId, Drive], Awaitable[str | None]], proxy_url: str, wire: Wire, store: Store,
    log: Path, recall: str, record: Record, environment: Mapping[str, str],
) -> AsyncGenerator[Mind]:
    """The model for the whole conversation: the brain's process, the MCP server it reaches hands through, the stage that
    speaks for it from the wire, the keeper of its context, and what answers hands' side questions. The brain is given
    `brain_tools` beside `tools`: what it does with a shell, to find what they act on, and a skill saying when."""
    llm = config.llm
    spans = CallSpans()
    talk = conversation(llm.config_dir, SessionId(str(uuid4())))
    # The brain is behind hands' proxy, so its usage is read off the wire, and it has a tool to ask it with.
    usage = Usage(talk.session)
    tools = [*tools, *brain_tools, audited(usage_tool(usage), record)]
    server = await serve_mcp(tools, record, spans)
    try:
        station = Station(llm.config_dir, workdir(llm.config_dir), llm.model, proxy_url, environment)
        try:
            brain = await start_brain(Launch(station, llm.account, brain_instruction(log, llm.config_dir, recall, config.personality), server.config(), talk), record)
        except Unstartable as error:
            # hands runs on no brain it could not start: its start is refused, saying why.
            raise CannotStart(str(error)) from error
        try:
            # [LAW:single-enforcer] everything hands asks in the background is asked here, of a Claude Code of
            # its own: nothing but the user's turns and their stops is ever typed into the brain.
            asides = Asides(station, record)
            stage = BrainStage(brain, tools, tail, refocus, handed, front, modality, opened, record, spans)
            keeper = Keeper(brain.session, partial(asides.ask, AsideKind.LINE, within=LINE_TIME), store, EVERY, record)
            with wire.joined(Kept(stage, keeper, asides, brain, usage)):
                watches = (Watch("the brain", lambda: outlived(brain)), Watch("the brain's turns", stage.ask_each), Watch("the brain's context", keeper.keep_asking))
                # A summary is as long as its Claude Code makes it: only its time is the summary's own.
                # The brain's stage notes each turn of the user's itself, as it types it.
                yield Mind(stage, watches, lambda kind, instruction, timeout: aside(partial(asides.ask, kind), instruction, timeout))
        finally:
            await brain.stop()
    finally:
        await server.close()


async def outlived(brain: Brain) -> None:
    # [LAW:no-silent-failure] a brain that ends while hands runs leaves every question unanswered, so it stops the run.
    code = await brain.exited()
    raise RuntimeError(f"the brain exited ({code}) while hands was running")


async def run(
    configure: Callable[[Mapping[str, str]], Configured], survey: Callable[[Configured | CannotStart], None], home: Home, heart: heartbeat.Heart, record: Record,
    degraded: Callable[[], tuple[heartbeat.Degradation, ...]], quit_event: asyncio.Event, after_crash: bool,
    environment: Mapping[str, str],
    run_start: Start,
    own: OwnModel,
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
        # Before the hooks are served: a turn that finishes while the models load is named once they have.
        names = Names()
        sessions = Sessions(
            permission_deadline=PERMISSION_DEADLINE_SECONDS, clock=time.monotonic, record=record, changes=deltas, names=names,
            typist=partial(type_into, environment), keyboards=partial(tmux.keyboards, environment=environment),
        )
        hooks = await serve_hooks(home, sessions, names, record)
        held.push_async_callback(hooks.cleanup)
        run_start.heard(hooks=home.socket)
        wire = Wire(wire_to(record))
        proxy = await serve_proxy(UPSTREAM, wire.observe, wire.route, clock=time.time)
        held.push_async_callback(proxy.close)
        # The url a Claude Code process's ANTHROPIC_BASE_URL is set to.
        run_start.heard(proxy=proxy.url, upstream=UPSTREAM)
        # [LAW:one-source-of-truth] the working sessions' exchanges reach the same observer as the brain's, so the log and
        # whatever listens hear one wire; and what they say of a turn reaches the registry as it is heard, as a hook does.
        def tapped(observed: Observed) -> None:
            wire.observe(observed)
            for move in moves(observed):
                sessions.hear(move)

        tap = await serve_tap(home.wire, tapped, record, clock=time.time)
        held.push_async_callback(tap.close)
        run_start.heard(tap=home.wire)
        display = await serve_display(sessions, DISPLAY_HOST, DISPLAY_PORT, DISPLAY_PATH, record)
        held.push_async_callback(display.cleanup)
        # [LAW:one-source-of-truth] the address the server bound, which the hooks' DISPLAY_URL must name.
        [(display_host, display_port, *_)] = display.addresses
        run_start.heard(display=f"http://{display_host}:{display_port}{DISPLAY_PATH}")
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
        # [LAW:one-source-of-truth] one owner of which trigger opens the user's turns at the desk: set_trigger switches
        # it, and the desk is driven by its edge.
        triggers = Triggers()
        # [LAW:one-source-of-truth] one queue of the cues owed to silence: the tools, the relay, and the turn's receipt owe
        # them, and the run plays them once its speaker is up and quiet.
        quiet_cues = QuietCues()
        # [LAW:one-source-of-truth] the one environment the run was handed: the settings' secrets, git's, and the brain's alike.
        config = await start(lambda: configured(lambda: configure(environment), survey, home, sessions, run_start), heart, sessions.live_count, degraded, quit_event)
        if config is not None:
            # [LAW:one-source-of-truth] one room of voices, told by the speaker model and named by the brain.
            room = Room(home.speakers, record)
            tools = [audited(tool, record) for tool in intermediary_tools(sessions, store, home, recounts, player, refocus, key.switch, triggers, config.wake, own, Catalogue(credentials(environment)), room, environment, lambda: quiet_cues.owe(WORKING))]
            # A session is started by the daemon, in the environment hands was started in, which is the user's, and is
            # started once the registry holds it, so the brain can stage for it at once; it is closed once the registry
            # no longer does.
            brain_tools = [
                audited(cued(start_session_tool(home, record, environment, sessions.live_members), lambda: quiet_cues.owe(WORKING)), record),
                audited(cued(close_session_tool(home, record, sessions), lambda: quiet_cues.owe(WORKING)), record),
            ]
            # [LAW:no-ambient-temporal-coupling] the model is up before the voice is built around its stage.
            async with mind(config, tools, brain_tools, lambda: as_sent(sessions, home), lambda: front_now(sessions, environment), lambda: key.modality, lambda: key.opened, refocus, handed_back(sessions), proxy.url, wire, store, home.audit, shlex.join(invocation(home, "recall")), record, environment) as minded:
                # What Whisper is primed with, read as each hold is transcribed.
                lexicon = Lexicon(sessions, home, environment, record)
                floor = Floor(lambda id: spoken_name(sessions, id), sessions.live_sessions)
                voice = await start(lambda: off_loop(lambda: build_voice(config, minded.llm, key, player, floor, lexicon, teller(home.speakers, room, record), record), "the voice load"), heart, sessions.live_count, degraded, quit_event)
                if voice is not None:
                    sentences = minded.summariser(AsideKind.SUMMARY, SENTENCE_INSTRUCTION, SENTENCES_TIMEOUT_SECONDS)
                    await converse(voice, home, sessions, heart, degraded, quit_event, after_crash, record, deltas, minded, store, sentences, names, recounts, quiet_cues, triggers, config.wake, run_start)
    return Ended(None if voice is None else _wall(voice.audio.output().sounded_at), sessions.live_count())


async def configured(configure: Callable[[], Configured], survey: Callable[[Configured | CannotStart], None], home: Home, sessions: Sessions, run_start: Start) -> VoiceConfig:
    """The configuration, once the sessions already running are listed and what hands is missing has been said;
    CannotStart once it has been said, where hands cannot run on the settings."""
    # A restart is back where it was before anything slow: every session with a file and a running process is listed.
    await sweep(home, sessions, frozenset())
    read = await off_loop(lambda: _attempted(configure), "the configuration read")
    # [LAW:single-enforcer] the backend is said as the configuration reached it, never its key or login read twice; and
    # [LAW:dataflow-not-control-flow] every step is said whether or not hands can start, so a start refused on its
    # settings still names each other step it is missing.
    await off_loop(lambda: survey(read), "the readiness check")
    if isinstance(read, CannotStart):
        raise read
    config = read.voice
    # [LAW:nothing-unseen] which settings won is read from the start's event, not re-derived from a shell: the file they came
    # from (None where the home has none and every setting is its default), the transcription server and collector they name, the
    # model the run reaches and the brain's account, the voice it
    # starts speaking in, the one the user kept or the default, the personality it comes across in, None for hands' own,
    # and the wake word the wake word trigger listens for, with the model of the user's own it is heard with, None for
    # openWakeWord's own.
    read_from = read.settings.path(home)
    run_start.heard(
        settings=read_from, transcription=config.transcription, collector=read.settings.config.collector,
        backend=type(config.llm).__name__, model=config.llm.model, account=config.llm.account, voice=config.voice,
        personality=config.personality, wake_word=config.wake.phrase, wake_word_model=str(config.wake.model) if isinstance(config.wake, Trained) else None,
    )
    return config


async def converse(
    voice: Voice,
    home: Home,
    sessions: Sessions,
    heart: heartbeat.Heart,
    degraded: Callable[[], tuple[heartbeat.Degradation, ...]],
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
    triggers: Triggers,
    wake_word: Word,
    run_start: Start,
) -> None:
    """Run the pipeline and what feeds it until the run is told to stop; raises what failed if anything did. The start is
    ready, and ends, once the pipeline has started."""
    pipeline = PipelineWatch(voice.worker)
    tails = Tails(sessions)
    channel = SystemChannel(voice.tts, post_notification, record)
    listen(voice, channel, after_crash)
    record_turns(voice.user_turns, voice.assistant_turns, record)
    cue_receipt(voice.user_turns, lambda: quiet_cues.owe(RECEIVED))
    failures: list[BaseException] = []

    def beat() -> None:
        # A turn opened with no microphone hears nothing, so it is not listening.
        deaf = voice.audio.deaf
        heart.beat(pipeline.state, _wall(voice.audio.output().sounded_at), sessions.live_count(), listening=voice.key.gate.turn_open and not deaf, degraded=(*((heartbeat.NO_MICROPHONE,) if deaf else ()), *degraded()))

    def stop_if_failed(task: asyncio.Task[None]) -> None:
        # [LAW:no-silent-failure] without the ticker nothing is denied at its deadline, without the sweep a dead
        # session stays listed, without the tail no record becomes a step, without the status reader no status Claude Code sets is heard, without the relay
        # nothing is asked aloud, without the progress player the focus is not heard working, without the narrator no finished turn or ended session is heard, without the summary store no backlog or unheard turn is ever said, without the namer no session is given a name, without the heartbeat the daemon looks dead while it runs, without the cue player a turn received and hands acting are never heard, without the device follower an unplugged headset leaves it deaf and mute, and without the talk key no turn starts, so any of
        # them failing stops the run where it can be seen: in its terminal, and as down to the shim and the indicator.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.opt(exception=error).error(f"{task.get_name()} failed; stopping")
            failures.append(error)
            quit_event.set()

    # First sight of every project a session is already working in: its backlog is said before anyone asks for it.
    for listing in sessions.live():
        store.want(listing.session.membership.cwd)
    overlays = Overlays(home)
    playing = Playing()
    utterances = Utterances(record)
    background = [
        asyncio.create_task(sessions.keep_time(TICK_SECONDS), name="the permission deadline ticker"),
        asyncio.create_task(keep_sweeping(home, sessions, SWEEP_SECONDS), name="the session liveness sweep"),
        asyncio.create_task(keep_tailing(tails, TAIL_SECONDS, sessions.apply), name="the transcript tail"),
        asyncio.create_task(keep_reading_statuses(sessions.live_ids, sessions.live_session, sessions.now, STATUS_SECONDS, sessions.apply), name="the status reader"),
        asyncio.create_task(utterances.keep(), name="the utterances"),
        asyncio.create_task(
            relay(sessions, utterances, voice.worker.queue_frame, partial(attending, home, overlays, lambda: attention(home)), lambda progress, amount, utterance: playing.put_nowait((progress, amount, utterance)), lambda: quiet_cues.owe(WORKING)),
            name="the session speech relay",
        ),
        asyncio.create_task(
            keep_playing(playing, sessions.live_session, voice.worker.queue_frame, minded.summariser(AsideKind.EXPLANATION, EXPLAIN_INSTRUCTION, EXPLAIN_TIMEOUT_SECONDS)),
            name="the progress player",
        ),
        asyncio.create_task(narrate(sessions, utterances, tails, voice.worker.queue_frame, lambda: attention(home), overlays, recounts, changes=deltas), name="the session narrator"),
        asyncio.create_task(keep_summarising(store, sentences, record), name="the summary store"),
        asyncio.create_task(keep_naming(names, sessions.live_members, minded.summariser(AsideKind.NAME, NAME_INSTRUCTION, NAME_TIMEOUT_SECONDS), record), name="the namer"),
        asyncio.create_task(keep_beating(beat, heart.period.total_seconds()), name="the heartbeat"),
        *(asyncio.create_task(watch.run(), name=watch.name) for watch in minded.watches),
    ]
    following = asyncio.create_task(follow_default_devices(pipeline.started, voice.audio, channel.say, record), name="the audio device follower")
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

    async def at_desk(move: Move, by: Trigger) -> None:
        match voice.key.move(move, by):
            case None:
                # A Shift at the desk while the turn is the phone's: nothing of it is cued, said, or beaten.
                return
            case taken:
                after_move(taken)
                for fact in told(taken, voice.audio.devices):
                    await channel.say(fact)

    async def drive_desk_once_started() -> None:
        # [LAW:no-ambient-temporal-coupling] a move reads the devices, which are known once the pipeline has opened
        # its streams; the desk is driven from then on.
        await pipeline.started.wait()
        await triggers.drive(drive_desk)

    async def drive_desk(trigger: Trigger) -> None:
        # [LAW:one-type-per-behavior] the trigger in use picks the edge that drives the desk: each trigger is one arm.
        match trigger:
            case "held key":
                await drive_talk_key(lambda move: at_desk(move, "held key"))
            case "engaged conversation":
                # Loaded as the trigger is switched to, in about half a second, and let go of as it is switched from.
                async with loaded(voice.audio.input().sample_rate, record) as ears:
                    await drive_engaged(tapped, voice.audio.input().overheard, ears, lambda move: at_desk(move, "engaged conversation"), record)
            case "wake word":
                # As engaged conversation's, with the wake word's model beside them; the talk key is not read.
                async with loaded(voice.audio.input().sample_rate, record) as ears, wake.loaded(voice.audio.input().sample_rate, home.wake_word, wake_word, record) as word:
                    woken = wake.listening(word, lambda: voice.audio.output().hands_speaking, record)
                    await drive(WAKE, untapped, voice.audio.input().overheard, ears, woken, lambda move: at_desk(move, "wake word"), record)

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
                phone_tasks.create_task(serve_phone(voice.phone, lambda text: voice.worker.queue_frame(Typed(text=text)), home, record), name="the phone's page")
                phone_tasks.create_task(cue_the_phone(), name="the phone's cues")
        finally:
            await voice.phone.stop()

    async def cue_silence_once_started() -> None:
        # As for the talk key: a cue is played on a stream the pipeline has opened, so what is owed before then waits.
        await pipeline.started.wait()
        await quiet_cues.keep_playing(voice.audio.output(), record)

    async def ready_once_started() -> None:
        await pipeline.started.wait()
        run_start.ended(record, None)

    readying = asyncio.create_task(ready_once_started(), name="the start's end")
    readying.add_done_callback(stop_if_failed)
    background.append(readying)
    cueing = asyncio.create_task(cue_silence_once_started(), name="the cues for silence")
    cueing.add_done_callback(stop_if_failed)
    background.append(cueing)
    talk_key = asyncio.create_task(drive_desk_once_started(), name="the desk's trigger")
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


