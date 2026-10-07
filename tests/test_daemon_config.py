"""The process boundary: the settings file, parsed once, and the secret each backend it names reaches its model with."""

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from conftest import onboard
from hands.daemon import cli, config, run
from hands.daemon.config import ANTHROPIC_MODEL, CLAUDE_MODELS, Claude, Config
from hands.daemon.starting import CannotStart, Ended, Start, start
from hands.daemon.backend import backend
from hands.sessions import audit, heartbeat, wrapper
from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.hookconfig import DISPLAY_PATH
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.sessions.wide import WideEvent
from hands.core.wire import UPSTREAM
from hands.voice import voices
from hands.voice.wakeword import Pretrained, Trained
from hands.voice.backends import Account, ClaudeCodeBackend

HOME = Home(Path("/Users/someone/.hands"))


def test_no_file_is_the_brain_on_sonnet(tmp_path: Path) -> None:
    assert config.load(Home(tmp_path)) == config.Settings(None, Config(llm=Claude(model=ANTHROPIC_MODEL), collector=None))
    assert config.parse("") == Config()


def test_the_file_names_the_backend_and_its_model(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-opus-5-5"\n')
    settings = config.load(home)
    assert settings.config == Config(llm=Claude(model="claude-opus-5-5"))
    assert settings.path(home) == home.config
    assert config.parse('[llm]\nmodel = "claude-opus-5-5"\n').llm == Claude(model="claude-opus-5-5")


def test_the_file_names_how_hands_comes_across_in_the_users_words(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    home.config.write_text('[talk]\npersonality = """\n  Dry and wry.\n"""\n')
    assert run.configured_from(home, config.load(home), os.environ).voice.personality == "Dry and wry."
    # Left out, hands comes across as its own.
    assert config.parse("").personality is None


def test_the_file_names_the_wake_word_one_of_openwakewords_own_or_the_users_own_model() -> None:
    # Left out, it is Hey Jarvis; one of openWakeWord's own is matched however it is capitalised and spaced.
    assert config.parse("").wake == Pretrained("Hey Jarvis")
    assert config.parse('[talk]\nwake_word = "hey  mycroft"\n').wake == Pretrained("Hey Mycroft")
    assert config.parse('[talk]\nwake_word = "ALEXA"\n').wake == Pretrained("Alexa")
    # The user's own is said as written, spaced as one of openWakeWord's is, and its model's path may start at the home
    # directory.
    assert config.parse('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "~/wake/hey_computer.onnx"\n').wake == Trained("Hey Computer", Path.home() / "wake" / "hey_computer.onnx")
    assert config.parse('[talk]\nwake_word = " Hey   Computer "\nwake_word_model = "/wake/hey_computer.onnx"\n').wake == Trained("Hey Computer", Path("/wake/hey_computer.onnx"))


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("[llm\n", "not TOML"),
        ('[llm]\nbackend = "local"\n', "backend 'local' is not one hands runs on: it runs on claude, the brain"),
        # A model API reached with a key is no harness hands drives.
        ('[llm]\nbackend = "anthropic"\n', "backend 'anthropic' is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key"),
        ('[llm]\nbackend = "openai"\n', "backend 'openai' is not one hands runs on"),
        # A url for the brain would be ignored, its requests going through hands' proxy, so it is refused.
        ('[llm]\nbackend = "claude"\nurl = "http://localhost:8080"\n', "[llm] for claude has no 'url'; it takes backend, model"),
        ('[llm]\nurl = "https://api.anthropic.com"\n', "[llm] for claude has no 'url'; it takes backend, model"),
        # A key misspelled is refused, not a setting silently left at its default.
        ('[llm]\nmodle = "m"\n', "has no 'modle'"),
        ('voice = "charles"\n', "the file has no 'voice'"),
        ("[llm]\nmodel = 4\n", "model should be a non-empty string, got 4"),
        ('llm = "claude"\n', "llm should be a table"),
        ('[talk]\npersonality = "  "\n', "[talk] personality should be a non-empty string"),
        ('[talk]\nmood = "dry"\n', "[talk] has no 'mood'; it takes personality, wake_word, wake_word_model"),
        # A phrase openWakeWord has no model of could never be heard, so it is refused, naming those it has.
        ('[talk]\nwake_word = "Hey Computer"\n', "[talk] wake_word 'Hey Computer' is not one of openWakeWord's own: Hey Jarvis, Hey Mycroft, Hey Rhasspy, Alexa"),
        ('[talk]\nwake_word = " "\n', "[talk] wake_word should be a non-empty string"),
        ('[talk]\nwake_word_model = "/models/hey_computer.onnx"\n', "wake_word_model needs wake_word"),
        ('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "models/hey_computer.onnx"\n', "is not a full path to an ONNX model"),
        ('[talk]\nwake_word = "Hey Computer"\nwake_word_model = "/models/hey_computer.tflite"\n', "is not a full path to an ONNX model"),
    ],
)
def test_a_file_that_does_not_parse_is_refused_saying_what_is_wrong(text: str, said: str) -> None:
    with pytest.raises(Rejected) as refused:
        config.parse(text)
    assert said in str(refused.value)


def test_a_refused_file_is_refused_naming_itself(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "local"\n')
    with pytest.raises(Rejected, match=f"{home.config}: \\[llm\\] backend 'local'"):
        config.load(home)


def test_the_collector_is_an_http_address_spelled_without_a_trailing_slash() -> None:
    assert config.parse("").collector is None
    assert config.parse('[telemetry]\ncollector = "http://otel.example:4318/"\n').collector == "http://otel.example:4318"
    assert config.parse('[telemetry]\ncollector = "https://[::1]:4318/otel"\n').collector == "https://[::1]:4318/otel"
    # No scheme, a bracket left open, no host, a port out of range, the traces endpoint itself, a query, credentials
    # urllib would take for part of the host and the log would hold: none is a
    # base address a batch could ever be posted under, so each is refused as the file is read, not batch by batch.
    for unusable in ("otel.example:4317", "http://[::1:4318", "http://:4318", "http://otel.example:99999", "http://otel.example:4318/v1/traces", "http://otel.example:4318?x=1", "http://user:secret@otel.example:4318"):
        with pytest.raises(Rejected, match="is not an OTLP/HTTP collector's base address") as refused:
            config.parse(f'[telemetry]\ncollector = "{unusable}"\n')
        # A refused edit is a log line: the address it refused is not in it.
        assert "secret" not in str(refused.value)
    with pytest.raises(Rejected, match="\\[telemetry\\] has no 'endpoint'"):
        config.parse('[telemetry]\nendpoint = "http://otel.example:4318"\n')


def test_a_file_naming_a_transcription_server_or_a_whisper_model_is_refused() -> None:
    # Either would be a setting silently not applied: hands transcribes with its own Whisper, on one model.
    with pytest.raises(Rejected, match="the file has no 'transcription'"):
        config.parse('[transcription]\nurl = "http://127.0.0.1:8610/v1"\n')
    with pytest.raises(Rejected, match="the file has no 'whisper'"):
        config.parse('[whisper]\nmodel = "mlx-community/whisper-large-v3-turbo"\n')


def test_a_brain_it_cannot_reach_stops_the_start_naming_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    home = Home(tmp_path / ".hands")
    with pytest.raises(CannotStart, match="`hands login` gives it one"):
        run.configured_from(home, config.load(home), os.environ)


def test_a_hands_setting_left_in_the_environment_stops_the_start_naming_the_file(tmp_path: Path) -> None:
    # The variables settings used to be: one still exported would run hands on the default backend, silently.
    home = Home(tmp_path)
    with pytest.raises(CannotStart, match=f"^HANDS_LLM, HANDS_WHISPER_MODEL set, .* settings go in {home.config}"):
        run.configured_from(home, config.load(home), {"HANDS_HOME": str(tmp_path), "HANDS_LLM": "claude", "HANDS_WHISPER_MODEL": "w"})


def test_claude_is_the_brain_on_the_login_in_hands_own_config_dir_with_no_key(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    assert backend(Claude(), home, os.environ) == ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=home.brain, account=Account("claude.ai", "brain@example.com"))
    assert backend(Claude(model="claude-opus-5-5"), home, os.environ).model == "claude-opus-5-5"


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_claude_runs_on_each_model_on_offer(fake_claude: Path, tmp_path: Path, model: str) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    assert backend(Claude(model=model), home, os.environ).model == model


def test_a_claude_model_not_on_offer_is_refused_as_the_file_is_parsed() -> None:
    # A model by its spoken name, one that does not exist, and a real one hands does not offer: each is refused naming the
    # four.
    for model in ("opus", "claude-opus-9", "claude-sonnet-5"):
        with pytest.raises(Rejected, match=f"^hands runs Claude on {', '.join(CLAUDE_MODELS)}, not {model}$"):
            config.parse(f'[llm]\nbackend = "claude"\nmodel = "{model}"\n')


def test_a_model_flag_outranks_the_files_on_the_files_backend(tmp_path: Path) -> None:
    home = Home(tmp_path)
    assert config.load(home, "claude-opus-5-5") == config.Settings(None, Config(llm=Claude(model="claude-opus-5-5")), "claude-opus-5-5")
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-haiku-4-5-20251001"\n[telemetry]\ncollector = "http://otel.example:4318"\n')
    assert config.load(home, "claude-opus-5-5").config == Config(llm=Claude(model="claude-opus-5-5"), collector="http://otel.example:4318")
    # The file's model, outranked, is never read: one hands no longer offers is no reason to refuse the run.
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-sonnet-4"\n')
    assert config.load(home, "claude-opus-5-5").config == Config(llm=Claude(model="claude-opus-5-5"))


def test_a_model_flag_hands_cannot_run_on_is_refused_naming_the_flag_not_the_file(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    with pytest.raises(Rejected) as refused:
        config.load(home, "opus")
    assert str(refused.value) == f"--model 'opus': hands runs Claude on {', '.join(CLAUDE_MODELS)}, not opus"


def test_a_blank_model_flag_is_refused_as_the_command_line_is_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--home", str(tmp_path), "run", "--model", "  "])
    assert exited.value.code == 2
    assert f"argument --model: a model's id, such as {ANTHROPIC_MODEL}, not '  '" in capsys.readouterr().err


def test_a_start_says_the_model_flag_it_was_given_though_refused_before_the_settings_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: False)
    monkeypatch.setattr(talkkey, "ask", lambda: None)
    for given in ((), ("--model", " claude-opus-5-5 ")):
        assert cli.main(["--home", str(home.root), "run", *given]) == 1
    capsys.readouterr()
    starts = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert [(start["outcome"], start["facts"]["model_flag"]) for start in starts] == [("failed", None), ("failed", "claude-opus-5-5")]


def test_a_model_flag_hands_cannot_run_on_refuses_the_start_at_the_door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hands.voice import talkkey

    home = Home(tmp_path / ".hands")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    assert cli.door(home, Start(restarted=False), "claude-opus-5-5") == config.load(home, "claude-opus-5-5")
    with pytest.raises(CannotStart, match="^--model 'opus': "):
        cli.door(home, Start(restarted=False), "opus")


@pytest.mark.parametrize(("given", "loads", "carried"), [((), None, ()), (("--model", " -local-model "), "-local-model", ("--model=-local-model",))])
def test_a_restart_runs_again_on_the_model_flag_the_run_was_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: tuple[str, ...], loads: str | None, carried: tuple[str, ...]) -> None:
    home = Home(tmp_path)
    loaded: list[str | None] = []

    def door(_home: Home, _start: Start, model: str | None) -> config.Settings:
        # Only what the restart is given is weighed here: an id that begins with a dash is no model hands offers.
        loaded.append(model)
        return config.Settings(None, Config(), model)

    class Again(Exception):
        pass

    def again(argv: list[str]) -> None:
        raise Again(argv)

    def hold(_home: Home) -> None:
        pass

    def run_here(*_arguments: object) -> tuple[str, int]:
        return "restart", 42

    monkeypatch.setattr(cli, "hold", hold)
    monkeypatch.setattr(cli, "door", door)
    monkeypatch.setattr(cli, "run_here", run_here)
    monkeypatch.setattr(cli, "again", again)
    with pytest.raises(Again) as restarted:
        cli.main(["--home", str(home.root), "run", *given])
    assert loaded == [loads]
    assert restarted.value.args[0][-2 - len(carried):] == ["--restarted", "42", *carried]
    # The restart's own command line parses back to the run it restarts, an id that begins with a dash and all.
    with pytest.raises(Again):
        cli.main(["--home", str(home.root), "run", *restarted.value.args[0][-2 - len(carried):]])
    assert loaded == [loads, loads]


async def test_an_edit_to_the_files_model_is_no_edit_while_a_model_flag_outranks_it_and_one_to_anything_else_is(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    running = config.load(home, "claude-opus-5-5")
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, _reachable, running, period=0.01), 0.3))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-haiku-4-5-20251001"\n')
    await asyncio.sleep(0.05)
    # Nor is one to a model hands does not offer: outranked, it is never read, so nothing is refused.
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "opus"\n')
    with pytest.raises(TimeoutError):
        await watching
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, _reachable, running, period=0.01), 2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\nmodel = "claude-haiku-4-5-20251001"\n[talk]\npersonality = "Dry."\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


def test_a_brain_that_would_load_its_accounts_skills_stops_the_run_naming_the_switches(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    (home.brain / "settings.json").unlink()
    with pytest.raises(Rejected, match="settings could not be read"):
        backend(Claude(), home, os.environ)
    (home.brain / "settings.json").write_text('{"syncClaudeAiPlugins": false}')
    with pytest.raises(Rejected, match='its account\'s syncClaudeAiSkills: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in'):
        backend(Claude(), home, os.environ)


def test_a_hands_built_without_its_fritter_stops_the_run_naming_the_rebuild(fake_claude: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    monkeypatch.setattr(wrapper, "PACKAGED", tmp_path / "package" / "bin" / "fritter")
    with pytest.raises(Rejected, match=r"hands' package carries no fritter at .*/package/bin/fritter: install hands again"):
        backend(Claude(), home, os.environ)


def test_a_brain_that_would_start_on_claude_codes_first_screens_stops_the_run_naming_hands_login(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain, trusted=False)
    with pytest.raises(Rejected, match=r"first screens \(.*/cwd untrusted\): `hands login` answers them"):
        backend(Claude(), home, os.environ)


def test_a_brain_with_no_login_stops_the_run_before_the_voice_loads_naming_the_command(monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(Rejected, match="`hands login` gives it one"):
        backend(Claude(), HOME, os.environ)


def _starting(tmp_path: Path) -> tuple[Home, Sessions, heartbeat.Heart, run.VoiceConfig]:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01))
    sessions = Sessions(permission_deadline=60.0, clock=time.monotonic, record=lambda _event: None)
    config = run.VoiceConfig(llm=ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=tmp_path / "brain", account=Account("claude.ai", "brain@example.com")), voice=voices.DEFAULT, personality="Dry and wry.", wake=Trained("Hey Computer", Path("/wake/hey_computer.onnx")))
    return Home(tmp_path), sessions, heart, config


async def test_the_start_beats_while_the_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A login check answered slowly is a start that is waiting, never one that looks stuck.
    home, sessions, heart, config = _starting(tmp_path)
    answered = threading.Event()

    def prompted() -> run.Configured:
        answered.wait()
        return run.Configured(config, run.Settings(b"", Config(collector="http://otel.example:4318")))

    run_start = Start(restarted=False)
    # A collector already failing while hands starts is said by the start's beats, as it is once hands runs.
    exporting_down = heartbeat.Degradation("can't export traces", "the collector has not taken all of its traces")
    starting = asyncio.create_task(start(lambda: run.configured(prompted, lambda _: None, home, sessions, run_start), heart, sessions.live_count, lambda: (exporting_down,), asyncio.Event()))
    # The prompt is answered only once the start has said "starting" three times while it waited.
    beats: set[datetime] = set()
    while len(beats) < 3:
        status = heartbeat.read(heart.path)
        if status is not None and status.pipeline == "starting":
            assert status.degraded == (exporting_down,)
            beats.add(status.written_at)
        await asyncio.sleep(0.005)
    answered.set()
    assert await starting == config
    # The start's event says which file the settings came from, the collector they name, which server and
    # model the run reaches and on what account, the voice it speaks in, the personality it comes across in, and the
    # wake word it listens for, with the model of the user's own it is heard with.
    recorded: list[Entry] = []
    run_start.ended(recorded.append, None)
    [event] = recorded
    assert isinstance(event, WideEvent) and (event.event, event.outcome) == ("hands.start", "ok")
    chosen = {name: event.facts[name] for name in ("settings", "collector", "backend", "base_url", "model", "account", "voice", "personality", "wake_word", "wake_word_model")}
    assert chosen == {
        "settings": home.config, "collector": "http://otel.example:4318",
        "backend": "ClaudeCodeBackend", "base_url": UPSTREAM, "model": ANTHROPIC_MODEL, "account": Account("claude.ai", "brain@example.com"), "voice": voices.DEFAULT,
        "personality": "Dry and wry.", "wake_word": "Hey Computer", "wake_word_model": "/wake/hey_computer.onnx",
    }


async def test_a_stop_during_the_configuration_read_ends_the_start(tmp_path: Path) -> None:
    home, sessions, heart, _config = _starting(tmp_path)
    never = threading.Event()

    def prompted() -> run.Configured:
        never.wait()
        raise AssertionError("the prompt was never answered")

    quit_event = asyncio.Event()
    quit_event.set()
    assert await start(lambda: run.configured(prompted, lambda _: None, home, sessions, Start(restarted=False)), heart, sessions.live_count, lambda: (), quit_event) is None
    never.set()


def test_a_refused_configuration_stops_the_start(tmp_path: Path) -> None:
    # Run as the CLI runs it, so only asyncio.run's caller sees it.
    home, sessions, heart, _config = _starting(tmp_path)

    def refused() -> run.Configured:
        raise CannotStart("no key")

    surveyed: list[run.Configured | CannotStart] = []
    with pytest.raises(CannotStart, match="no key"):
        asyncio.run(start(lambda: run.configured(refused, surveyed.append, home, sessions, Start(restarted=False)), heart, sessions.live_count, lambda: (), asyncio.Event()))
    # Every other step is still said: the readiness check is given the refusal, as the backend's step.
    [refusal] = surveyed
    assert isinstance(refusal, CannotStart) and str(refusal) == "no key"


def test_a_restart_lists_its_running_sessions_before_the_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, sessions, heart, config = _starting(tmp_path)
    order: list[str] = []

    async def swept(*_: object) -> None:
        order.append("sweep")

    def configure() -> run.Configured:
        order.append("configure")
        return run.Configured(config, run.Settings(b"", Config()))

    monkeypatch.setattr(run, "sweep", swept)
    asyncio.run(start(lambda: run.configured(configure, lambda _: order.append("survey"), home, sessions, Start(restarted=False)), heart, sessions.live_count, lambda: (), asyncio.Event()))
    assert order == ["sweep", "configure", "survey"]


async def test_a_start_says_where_the_run_listens_as_it_serves_each(monkeypatch: pytest.MonkeyPatch) -> None:
    # A unix socket path is capped near 104 bytes on macOS, so not under pytest's long tmp_path.
    root = Path(tempfile.mkdtemp(prefix="hands-"))
    home = Home(root)
    # The system picks the display's port, so a hands running on this machine keeps its own.
    monkeypatch.setattr(run, "DISPLAY_PORT", 0)
    heart = heartbeat.Heart(root / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01))

    def refused(_environment: Mapping[str, str]) -> run.Configured:
        raise CannotStart("no key")

    run_start = Start(restarted=False)
    recorded: list[Entry] = []
    try:
        with pytest.raises(CannotStart, match="no key"), run_start.ending(recorded.append):
            await run.run(refused, lambda _: None, home, heart, recorded.append, lambda: (), asyncio.Event(), False, {}, run_start, config.OwnModel(home, _NO_FILE, _reachable))
    finally:
        shutil.rmtree(root)
    # Refused at its settings, after every server was up: the start says where each listened.
    [event] = [entry for entry in recorded if isinstance(entry, WideEvent) and entry.event == "hands.start"]
    proxy, display = event.facts["proxy"], event.facts["display"]
    assert (event.outcome, event.facts["hooks"], event.facts["upstream"], event.facts["tap"]) == ("failed", home.socket, UPSTREAM, home.wire)
    assert isinstance(proxy, str) and re.fullmatch(r"http://127\.0\.0\.1:[1-9]\d*", proxy)
    # The port the display was bound on, not the 0 it was asked for.
    assert isinstance(display, str) and re.fullmatch(rf"http://127\.0\.0\.1:[1-9]\d*{DISPLAY_PATH}", display)


def test_a_start_refused_says_why_in_the_audit_log_and_in_hands_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Started from a launcher whose terminal nobody watches, the reason is still on record: the run's first read of its
    # configuration, through the CLI's start, refusing a setting left in the environment.
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    # No indicator is shown, and the test's own log sinks are kept.
    def unshown(_home: Home) -> int:
        return 0

    def kept(*_: object) -> None:
        pass

    monkeypatch.setattr(cli, "start_indicator", unshown)
    monkeypatch.setattr(cli, "reap", kept)
    monkeypatch.setattr(cli, "to_terminal", kept)
    monkeypatch.setattr(cli.logger, "remove", kept)

    def loaded(home: Home, settings: config.Settings, heart: heartbeat.Heart, record: audit.Record, _degraded: Callable[[], tuple[heartbeat.Degradation, ...]], _after_crash: bool, run_start: Start) -> cli.Run:
        sessions = Sessions(permission_deadline=60.0, clock=time.monotonic, record=record)

        async def refused(quit_event: asyncio.Event) -> Ended:
            configure = partial(run.configured_from, home, settings, {"HANDS_LLM": "claude"})
            await start(lambda: run.configured(configure, lambda _: None, home, sessions, run_start), heart, sessions.live_count, lambda: (), quit_event)
            raise AssertionError("a start with HANDS_LLM set went on")

        return refused

    monkeypatch.setattr(cli, "loaded", loaded)
    assert cli.main(["--home", str(home.root), "run"]) == 1
    reason = f"HANDS_LLM set, and hands reads no setting from the environment; settings go in {home.config}"
    assert capsys.readouterr().err == f"hands: {reason}\n"
    # The start's one event is failed with the reason, and says which run it was.
    [refused] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert (refused["level"], refused["outcome"], refused["error"]) == ("error", "failed", f"CannotStart: {reason}")
    assert (refused["facts"]["pid"], refused["facts"]["restarted"], refused["facts"]["after_crash"]) == (os.getpid(), False, False)
    assert cli.main(["--home", str(home.root), "status"]) == 1
    assert capsys.readouterr().out == f"hands refused to start 0s ago: {reason}\n"


def test_a_start_that_fails_before_its_launch_is_still_one_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Past the door, before the run's launch: the indicator cannot be started.
    from hands.voice import talkkey

    home = Home(tmp_path)
    monkeypatch.setattr(talkkey, "granted", lambda: True)

    def unshown(_home: Home) -> int:
        raise OSError("no indicator")

    def kept(*_: object) -> None:
        pass

    monkeypatch.setattr(cli, "start_indicator", unshown)
    monkeypatch.setattr(cli, "to_terminal", kept)
    monkeypatch.setattr(cli.logger, "remove", kept)
    with pytest.raises(OSError, match="no indicator"):
        cli.main(["--home", str(home.root), "run"])
    [failed] = [line for line in map(json.loads, audit.tail(home.audit, 100)[0]) if line.get("event") == "hands.start"]
    assert (failed["outcome"], failed["error"]) == ("failed", "OSError: no indicator")


def test_a_settings_file_it_cannot_read_refuses_the_start_at_the_door_and_a_restarts_in_its_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Before its first heartbeat a run holds none: the one there, here a crash's, is left for the next run to read.
    from hands.voice import talkkey

    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "nonesuch"\n')
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    gone = subprocess.Popen(["true"])
    gone.wait()
    crashed = heartbeat.Heart(home.status, gone.pid, datetime.now(UTC), heartbeat.HEARTBEAT)
    crashed.beat("running", None, 0, listening=False, degraded=())
    assert cli.main(["--home", str(home.root), "run"]) == 1
    reason = capsys.readouterr().err.removeprefix("hands: ").rstrip("\n")
    assert str(home.config) in reason
    # Refused at the door, before the settings name a collector: the start's event is on the log alone.
    assert [(line["event"], line["error"]) for line in map(json.loads, audit.tail(home.audit, 10)[0])] == [("hands.start", f"CannotStart: {reason}")]
    assert cli.crashed_before(home)
    # A restart's run holds the heartbeat from the outset, the run before it having beat starting under its pid.
    assert cli.main(["--home", str(home.root), "run", "--restarted", "0"]) == 1
    capsys.readouterr()
    assert cli.main(["--home", str(home.root), "status"]) == 1
    assert capsys.readouterr().out == f"hands refused to start 0s ago: {reason}\n"


def test_a_home_whose_fritter_is_not_the_one_hands_carries_refuses_the_start_at_the_door_naming_install_fritter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # As after updating hands: the home's copy is the fritter an older hands carried, which may not know what this one asks.
    from hands.voice import talkkey

    home = Home(tmp_path)
    home.bin.mkdir()
    home.fritter.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    assert cli.main(["--home", str(home.root), "run"]) == 1
    reason = capsys.readouterr().err.removeprefix("hands: ").rstrip("\n")
    assert f"{home.fritter} is not the fritter this hands carries, {wrapper.PACKAGED}" in reason
    assert reason.endswith("run `hands install-fritter`, then run hands again.")
    [refused] = [line for line in map(json.loads, audit.tail(home.audit, 10)[0]) if line["event"] == "hands.start"]
    assert (refused["error"], refused["facts"]["fritter"]) == (f"CannotStart: {reason}", "stale")
    # A restart's run, as follows updating a running hands, writes the refusal into the heartbeat `hands status` reads.
    assert cli.main(["--home", str(home.root), "run", "--restarted", "0"]) == 1
    capsys.readouterr()
    assert cli.main(["--home", str(home.root), "status"]) == 1
    assert capsys.readouterr().out == f"hands refused to start 0s ago: {reason}\n"


@pytest.mark.parametrize("copy", ["current", "absent", "unpackaged"])
def test_a_home_whose_fritter_is_the_one_hands_carries_or_none_passes_the_door(copy: wrapper.Copy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A hands built without its fritter passes too: the survey, and the brain at its launch, name the rebuild.
    from hands.voice import talkkey

    home = Home(tmp_path / ".hands")
    if copy == "current":
        home.bin.mkdir(parents=True)
        shutil.copy2(wrapper.PACKAGED, home.fritter)
    if copy == "unpackaged":
        home.bin.mkdir(parents=True)
        home.fritter.write_bytes(b"#!/bin/sh\n")
        monkeypatch.setattr(wrapper, "PACKAGED", tmp_path / "package" / "bin" / "fritter")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    run_start = Start(restarted=False)
    assert cli.door(home, run_start, None) == config.load(home)
    events: list[WideEvent] = []
    run_start.ended(events.append, None)
    assert [event.facts["fritter"] for event in events] == [copy]


def test_a_home_whose_fritter_cannot_be_read_refuses_the_start_at_the_door_saying_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A link to a fritter that is gone is a copy that cannot be read, never no copy at all.
    from hands.voice import talkkey

    home = Home(tmp_path / ".hands")
    home.bin.mkdir(parents=True)
    home.fritter.symlink_to(tmp_path / "gone")
    monkeypatch.setattr(talkkey, "granted", lambda: True)
    run_start = Start(restarted=False)
    with pytest.raises(CannotStart, match=f"cannot tell whether {re.escape(str(home.fritter))} is the fritter this hands carries: "):
        cli.door(home, run_start, None)
    events: list[WideEvent] = []
    run_start.ended(events.append, None)
    assert [event.facts["fritter"] for event in events] == ["unreadable"]


def test_the_voice_is_charles_until_one_is_chosen_and_the_chosen_one_after_a_restart(fake_claude: Path, tmp_path: Path) -> None:
    """Charles by the name the installed pocket_tts resolves itself, until the user chooses another, which the next run
    is built in.

    A voice state is a cache computed by one network and noise under another, and the loader does not check; so the
    default is a catalogue name, which the package maps to the state primed by the weights it loads. The membership
    check here is the one get_state_for_audio_prompt makes, so a dependency bump that drops the name fails in its commit.
    """
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # pyright: ignore[reportPrivateUsage]

    logged_in = os.environ
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    assert run.configured_from(home, config.load(home), logged_in).voice.voice == "charles"
    assert "charles" in _ORIGINS_OF_PREDEFINED_VOICES
    voices.keep(home, voices.parse_voice("Bill Boerst"))
    assert run.configured_from(home, config.load(home), logged_in).voice.voice == "bill_boerst"
    # A kept name the installed pocket_tts no longer has stops the start, naming the file to fix.
    home.voice.write_text("zed\n")
    with pytest.raises(CannotStart, match=f"{home.voice} says 'zed'"):
        run.configured_from(home, config.load(home), logged_in)
    # One it cannot read stops it the same way, naming the file.
    home.voice.unlink()
    home.voice.mkdir()
    with pytest.raises(CannotStart, match=str(home.voice)):
        run.configured_from(home, config.load(home), logged_in)


# The settings of a run started with no file: every default.
_NO_FILE = config.Settings(None, Config())


def _reachable(_settings: Config) -> None:
    pass


async def _edited_within(home: Home, recorded: list[Entry], seconds: float = 0.5) -> SettingsEdited | None:
    try:
        return await asyncio.wait_for(config.edited(home, recorded.append, _reachable, config.load(home), period=0.01), seconds)
    except TimeoutError:
        return None


async def test_an_edit_that_parses_is_heard_and_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_a_file_saved_unchanged_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert recorded == []


async def test_an_edit_that_does_not_parse_is_said_and_outlived_until_one_that_does(tmp_path: Path) -> None:
    home = Home(tmp_path)
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "local"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    assert recorded == [SettingsEdited(path=str(home.config), refused=f"{home.config}: [llm] backend 'local' is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key")]
    assert audit.level(recorded[0]) == "error"
    home.config.write_text('[llm]\nmodel = "claude-haiku-4-5-20251001"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert len(recorded) == 1


async def test_settings_removed_are_an_edit_back_to_the_defaults(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.unlink()
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_an_edit_undone_back_to_the_settings_the_run_is_on_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.5))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "local"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert [type(entry) for entry in recorded] == [SettingsEdited] and audit.level(recorded[0]) == "error"


async def test_a_file_that_cannot_be_read_is_a_refused_edit_and_outlived(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=2.0))
    await asyncio.sleep(0.05)
    home.config.unlink()
    home.config.mkdir()
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    assert recorded == [SettingsEdited(path=str(home.config), refused=f"{home.config} could not be read: [Errno 21] Is a directory: '{home.config}'")]
    home.config.rmdir()
    home.config.write_text('[llm]\nmodel = "claude-haiku-4-5-20251001"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


async def test_a_save_written_in_two_steps_is_weighed_once_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    # What each poll after the run's read reads, as an editor that truncates and then writes leaves it: one poll lands between the two.
    reads = iter([b'[llm]\nbackend = "cla', b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-opus-5-5"\n'])

    def held(_home: Home) -> bytes | None:
        return next(reads)

    monkeypatch.setattr(config, "_held", held)
    recorded: list[Entry] = []
    assert await config.edited(home, recorded.append, _reachable, _NO_FILE, period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_an_edit_naming_a_model_hands_cannot_reach_is_said_and_outlived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    home = Home(tmp_path / ".hands")
    onboard(home.brain)
    monkeypatch.setenv("LOGGED_IN", "0")
    recorded: list[Entry] = []
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, partial(cli.reachable, home), config.load(home), period=0.01), 2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    [refused] = recorded
    assert isinstance(refused, SettingsEdited) and refused.refused is not None and "`hands login` gives it one" in refused.refused
    monkeypatch.setenv("LOGGED_IN", "1")
    home.config.write_text('[llm]\nmodel = "claude-haiku-4-5-20251001"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


async def test_a_comment_added_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('# opus for now\n[llm]\nmodel = "claude-opus-5-5"\n')
    assert await watching is None
    assert recorded == []


async def test_an_edit_saved_over_while_it_is_weighed_is_not_taken(tmp_path: Path) -> None:
    home = Home(tmp_path)
    recorded: list[Entry] = []

    def overwritten(settings: Config) -> None:
        # The first edit is saved over with a typo while its backend is checked.
        if settings.llm == Claude(model="claude-opus-5-5"):
            home.config.write_text('[llm]\nbackend = "claud"\n')

    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, overwritten, config.load(home), period=0.01), 0.5))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nmodel = "claude-opus-5-5"\n')
    with pytest.raises(TimeoutError):
        await watching
    assert [type(entry) for entry in recorded] == [SettingsEdited] and "'claud'" in str(recorded[0])


async def test_an_edit_saved_over_while_weighed_and_back_again_is_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    edit, other = b'[llm]\nmodel = "claude-opus-5-5"\n', b'[llm]\nmodel = "claude-haiku-4-5-20251001"\n'
    # Seen, settled, saved over as its backend is checked, and back before the next poll.
    reads = iter([edit, edit, other, edit, edit])

    def held(_home: Home) -> bytes | None:
        return next(reads)

    monkeypatch.setattr(config, "_held", held)
    recorded: list[Entry] = []
    assert await config.edited(home, recorded.append, _reachable, _NO_FILE, period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []
