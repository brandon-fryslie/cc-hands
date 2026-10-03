"""The process boundary: the settings file, parsed once, and the secret each backend it names reaches its model with."""

import asyncio
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from hands.daemon import cli, config, run
from hands.daemon.config import ANTHROPIC_MODEL, ANTHROPIC_URL, OPENAI_MODEL, OPENAI_URL, Anthropic, Claude, Config, OpenAI
from hands.daemon.starting import start
from hands.daemon.run import backend
from hands.sessions import audit, heartbeat
from hands.sessions.audit import Entry, LLMChosen, SettingsEdited, SettingsRead, VoiceChosen, encoded
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.registry import Sessions
from hands.core.wire import UPSTREAM
from hands.voice import voices
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, OpenAICompatibleBackend

HOME = Home(Path("/Users/someone/.hands"))


def test_no_file_is_claude_on_the_anthropic_api_and_whisper_large_v3_turbo(tmp_path: Path) -> None:
    assert config.load(Home(tmp_path)) == (Config(llm=Anthropic(url=ANTHROPIC_URL, model=ANTHROPIC_MODEL), whisper_model="mlx-community/whisper-large-v3-turbo"), None)
    assert config.parse("") == Config()


def test_the_file_names_the_backend_its_server_and_model_and_the_whisper_model(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "openai"\nurl = "https://reseller.example/v1"\nmodel = "gpt-other"\n\n[whisper]\nmodel = "w"\n')
    assert config.load(home) == (Config(llm=OpenAI(url="https://reseller.example/v1", model="gpt-other"), whisper_model="w"), home.config)
    assert config.parse('[llm]\nbackend = "openai"\n').llm == OpenAI(url=OPENAI_URL, model=OPENAI_MODEL)
    assert config.parse('[llm]\nurl = "https://api-chicago.codexapi.pro"\nmodel = "claude-other"\n').llm == Anthropic(url="https://api-chicago.codexapi.pro", model="claude-other")
    assert config.parse('[llm]\nbackend = "claude"\nmodel = "claude-other"\n').llm == Claude(model="claude-other")


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("[llm\n", "not TOML"),
        ('[llm]\nbackend = "local"\n', "backend 'local' is not one of: anthropic, openai, claude"),
        ('[llm]\nbackend = "gpt"\n', "is not one of"),
        # A url for the brain would be ignored, its requests going through hands' proxy, so it is refused.
        ('[llm]\nbackend = "claude"\nurl = "http://localhost:8080"\n', "[llm] for claude has no 'url'; it takes backend, model"),
        ('[llm]\nurl = "http://localhost:8080/v1"\n', "ends in /v1"),
        ('[llm]\nurl = "https://api-chicago.codexapi.pro/v1/"\n', "ends in /v1"),
        # A key misspelled is refused, not a setting silently left at its default.
        ('[llm]\nmodle = "m"\n', "has no 'modle'"),
        ('voice = "charles"\n', "the file has no 'voice'"),
        ('[llm]\nurl = "  "\n', "url should be a non-empty string"),
        ("[llm]\nmodel = 4\n", "model should be a non-empty string, got 4"),
        ('llm = "claude"\n', "llm should be a table"),
    ],
)
def test_a_file_that_does_not_parse_is_refused_saying_what_is_wrong(text: str, said: str) -> None:
    with pytest.raises(Rejected) as refused:
        config.parse(text)
    assert said in str(refused.value)


def test_a_refused_file_stops_the_start_naming_itself(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "local"\n')
    with pytest.raises(SystemExit, match=f"{home.config}: \\[llm\\] backend 'local'"):
        run.configured_from(home, {"ANTHROPIC_API_KEY": "k"})


def test_a_backend_it_cannot_reach_stops_the_start_naming_what_is_missing(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "openai"\n')
    with pytest.raises(SystemExit, match="hands: OPENAI_API_KEY is not set"):
        run.configured_from(home, {})


def test_a_hands_setting_left_in_the_environment_stops_the_start_naming_the_file(tmp_path: Path) -> None:
    # The variables settings used to be: one still exported would run hands on the default backend, silently.
    home = Home(tmp_path)
    with pytest.raises(SystemExit, match=f"HANDS_LLM, HANDS_WHISPER_MODEL set, .* settings go in {home.config}"):
        run.configured_from(home, {"ANTHROPIC_API_KEY": "k", "HANDS_HOME": str(tmp_path), "HANDS_LLM": "claude", "HANDS_WHISPER_MODEL": "w"})


def test_anthropic_on_its_own_url_spelled_with_a_slash_is_its_own_api() -> None:
    assert config.parse('[llm]\nurl = "https://api.anthropic.com/"\n').llm == config.Anthropic()


def test_anthropic_on_its_own_api_is_keyed_by_the_environment_else_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    kept: dict[str, str] = {}
    monkeypatch.setattr(run, "keychain_password", kept.get)
    with pytest.raises(Rejected, match="ANTHROPIC_API_KEY is not set and the keychain holds no HANDS_LLM_ANT_KEY"):
        backend(Anthropic(), HOME, {})
    kept["HANDS_LLM_ANT_KEY"] = "from-keychain"
    assert backend(Anthropic(), HOME, {}) == AnthropicBackend(base_url=ANTHROPIC_URL, api_key="from-keychain", model=ANTHROPIC_MODEL)
    assert backend(Anthropic(), HOME, {"ANTHROPIC_API_KEY": "k"}) == AnthropicBackend(base_url=ANTHROPIC_URL, api_key="k", model=ANTHROPIC_MODEL)


def test_the_keychain_key_never_leaves_for_another_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "keychain_password", {"HANDS_LLM_ANT_KEY": "anthropic-own"}.get)
    other = Anthropic(url="https://api-chicago.codexapi.pro", model="claude-other")
    with pytest.raises(Rejected, match="ANTHROPIC_API_KEY is not set"):
        backend(other, HOME, {})
    assert backend(other, HOME, {"ANTHROPIC_API_KEY": "k"}) == AnthropicBackend(base_url="https://api-chicago.codexapi.pro", api_key="k", model="claude-other")


def test_openai_needs_its_key() -> None:
    # Unset, or an empty line in a .env, or only space, is no key.
    for environment in ({}, {"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": "   "}):
        with pytest.raises(Rejected, match="OPENAI_API_KEY is not set"):
            backend(OpenAI(), HOME, environment)
    # Space around a key in a .env line is not part of the key.
    assert backend(OpenAI(), HOME, {"OPENAI_API_KEY": " k "}) == OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="k", model=OPENAI_MODEL)
    assert backend(OpenAI(url="https://reseller.example/v1", model="gpt-other"), HOME, {"OPENAI_API_KEY": "k"}) == OpenAICompatibleBackend(
        base_url="https://reseller.example/v1", api_key="k", model="gpt-other"
    )


def test_claude_is_the_brain_on_the_login_in_hands_own_config_dir_with_no_key(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    home.brain.mkdir(parents=True)
    (home.brain / "settings.json").write_text('{"syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}')
    assert backend(Claude(), home, os.environ) == ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=home.brain, account="brain@example.com")
    assert backend(Claude(model="claude-other"), home, os.environ).model == "claude-other"


def test_a_brain_that_would_load_its_accounts_skills_stops_the_run_naming_the_switches(fake_claude: Path, tmp_path: Path) -> None:
    home = Home(tmp_path / ".hands")
    home.brain.mkdir(parents=True)
    with pytest.raises(Rejected, match="settings could not be read"):
        backend(Claude(), home, os.environ)
    (home.brain / "settings.json").write_text('{"syncClaudeAiPlugins": false}')
    with pytest.raises(Rejected, match='its account\'s syncClaudeAiSkills: set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in'):
        backend(Claude(), home, os.environ)


def test_the_brain_is_logged_as_reaching_anthropics_api_through_the_proxy_on_its_account() -> None:
    brain = ClaudeCodeBackend(model=ANTHROPIC_MODEL, config_dir=HOME.brain, account="brain@example.com")
    assert run._server(brain) == UPSTREAM  # pyright: ignore[reportPrivateUsage]
    assert run._account(brain) == "brain@example.com"  # pyright: ignore[reportPrivateUsage]


def test_a_brain_with_no_login_stops_the_run_before_the_voice_loads_naming_the_command(monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(Rejected, match="cd /Users/someone/.hands/brain/cwd && CLAUDE_CONFIG_DIR=/Users/someone/.hands/brain claude"):
        backend(Claude(), HOME, os.environ)


def test_a_backend_printed_does_not_print_its_key() -> None:
    """The eval prints the backend it runs on, and a key printed is a key leaked."""
    for printed in (
        OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="sk-secret", model=OPENAI_MODEL),
        AnthropicBackend(base_url=ANTHROPIC_URL, api_key="sk-secret", model=ANTHROPIC_MODEL),
    ):
        assert "sk-secret" not in repr(printed) and "sk-secret" not in str(printed)


def _starting(tmp_path: Path) -> tuple[Home, Sessions, heartbeat.Heart, run.VoiceConfig]:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01))
    sessions = Sessions(permission_deadline=60.0, clock=time.monotonic, record=lambda _event: None)
    config = run.VoiceConfig(llm=AnthropicBackend(base_url=ANTHROPIC_URL, api_key="sk-secret", model=ANTHROPIC_MODEL), whisper_model="w", voice=voices.DEFAULT)
    return Home(tmp_path), sessions, heart, config


async def test_the_start_beats_while_the_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A keychain prompt answered slowly is a start that is waiting, never one that looks stuck.
    home, sessions, heart, config = _starting(tmp_path)
    answered = threading.Event()

    def prompted() -> run.Configured:
        answered.wait()
        return run.Configured(config, home.config)

    recorded: list[Entry] = []
    starting = asyncio.create_task(start(lambda: run.configured(prompted, lambda: None, home, sessions, recorded.append), heart, sessions.live_count, asyncio.Event()))
    # The prompt is answered only once the start has said "starting" three times while it waited.
    beats: set[datetime] = set()
    while len(beats) < 3:
        status = heartbeat.read(heart.path)
        if status is not None and status.pipeline == "starting":
            beats.add(status.written_at)
        await asyncio.sleep(0.005)
    answered.set()
    assert await starting == config
    # The log says which file the settings came from and the Whisper model they name, which server and model the run
    # reaches, and never with what key, and the voice it speaks in.
    assert recorded == [SettingsRead(path=str(home.config), whisper_model="w"), LLMChosen(backend="AnthropicBackend", base_url=ANTHROPIC_URL, model=ANTHROPIC_MODEL, account=None), VoiceChosen(voice=voices.DEFAULT)]
    assert "sk-secret" not in str([encoded(entry) for entry in recorded])


async def test_a_stop_during_the_configuration_read_ends_the_start(tmp_path: Path) -> None:
    home, sessions, heart, _config = _starting(tmp_path)
    never = threading.Event()

    def prompted() -> run.Configured:
        never.wait()
        raise AssertionError("the prompt was never answered")

    quit_event = asyncio.Event()
    quit_event.set()
    assert await start(lambda: run.configured(prompted, lambda: None, home, sessions, lambda _event: None), heart, sessions.live_count, quit_event) is None
    never.set()


def test_a_refused_configuration_stops_the_start(tmp_path: Path) -> None:
    # Run as the CLI runs it: a SystemExit from a task leaves the event loop itself, so only asyncio.run's caller sees it.
    home, sessions, heart, _config = _starting(tmp_path)

    def refused() -> run.Configured:
        raise SystemExit("no key")

    with pytest.raises(SystemExit, match="no key"):
        asyncio.run(start(lambda: run.configured(refused, lambda: None, home, sessions, lambda _event: None), heart, sessions.live_count, asyncio.Event()))


def test_the_voice_is_charles_until_one_is_chosen_and_the_chosen_one_after_a_restart(tmp_path: Path) -> None:
    """Charles by the name the installed pocket_tts resolves itself, until the user chooses another, which the next run
    is built in.

    A voice state is a cache computed by one network and noise under another, and the loader does not check; so the
    default is a catalogue name, which the package maps to the state primed by the weights it loads. The membership
    check here is the one get_state_for_audio_prompt makes, so a dependency bump that drops the name fails in its commit.
    """
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # pyright: ignore[reportPrivateUsage]

    keyed = {"ANTHROPIC_API_KEY": "sk-test"}
    home = Home(tmp_path)
    assert run.configured_from(home, keyed).voice.voice == "charles"
    assert "charles" in _ORIGINS_OF_PREDEFINED_VOICES
    voices.keep(home, voices.parse_voice("Bill Boerst"))
    assert run.configured_from(home, keyed).voice.voice == "bill_boerst"
    # A kept name the installed pocket_tts no longer has stops the start, naming the file to fix.
    home.voice.write_text("zed\n")
    with pytest.raises(SystemExit, match=f"{home.voice} says 'zed'"):
        run.configured_from(home, keyed)
    # One it cannot read stops it the same way, naming the file.
    home.voice.unlink()
    home.voice.mkdir()
    with pytest.raises(SystemExit, match=str(home.voice)):
        run.configured_from(home, keyed)


def _reachable(_settings: Config) -> None:
    pass


async def _edited_within(home: Home, recorded: list[Entry], seconds: float = 0.5) -> SettingsEdited | None:
    try:
        return await asyncio.wait_for(config.edited(home, recorded.append, _reachable, period=0.01), seconds)
    except TimeoutError:
        return None


async def test_an_edit_that_parses_is_heard_and_said(tmp_path: Path) -> None:
    home = Home(tmp_path)
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_a_file_saved_unchanged_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\n')
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
    assert recorded == [SettingsEdited(path=str(home.config), refused=f"{home.config}: [llm] backend 'local' is not one of: anthropic, openai, claude")]
    assert audit.level(recorded[0]) == "error"
    home.config.write_text('[llm]\nbackend = "openai"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert len(recorded) == 1


async def test_settings_removed_are_an_edit_back_to_the_defaults(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded))
    await asyncio.sleep(0.05)
    home.config.unlink()
    assert await watching == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_an_edit_undone_back_to_the_settings_the_run_is_on_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.5))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "local"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    assert await watching is None
    assert [type(entry) for entry in recorded] == [SettingsEdited] and audit.level(recorded[0]) == "error"


async def test_a_file_that_cannot_be_read_is_a_refused_edit_and_outlived(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
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
    home.config.write_text('[llm]\nbackend = "openai"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


async def test_a_save_written_in_two_steps_is_weighed_once_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    # What each poll reads, as an editor that truncates and then writes leaves it: one poll lands between the two.
    reads = iter([None, b'[llm]\nbackend = "cla', b'[llm]\nbackend = "claude"\n', b'[llm]\nbackend = "claude"\n', b'[llm]\nbackend = "claude"\n'])

    def held(_home: Home) -> bytes | None:
        return next(reads)

    monkeypatch.setattr(config, "_held", held)
    recorded: list[Entry] = []
    assert await config.edited(home, recorded.append, _reachable, period=0) == SettingsEdited(path=str(home.config), refused=None)
    assert recorded == []


async def test_an_edit_naming_a_model_hands_cannot_reach_is_said_and_outlived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = Home(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recorded: list[Entry] = []
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, partial(cli.reachable, home), period=0.01), 2.0))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "openai"\n')
    while not recorded:
        await asyncio.sleep(0.01)
    assert not watching.done()
    assert recorded == [SettingsEdited(path=str(home.config), refused="OPENAI_API_KEY is not set; the [llm] backend hands is set to run on needs it to reach its model.")]
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    home.config.write_text('[llm]\nbackend = "openai"\nmodel = "gpt-other"\n')
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


async def test_a_comment_added_is_no_edit(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    recorded: list[Entry] = []
    watching = asyncio.create_task(_edited_within(home, recorded, seconds=0.2))
    await asyncio.sleep(0.05)
    home.config.write_text('# openai later\n[llm]\nbackend = "claude"\n')
    assert await watching is None
    assert recorded == []


async def test_an_edit_saved_over_while_it_is_weighed_is_not_taken(tmp_path: Path) -> None:
    home = Home(tmp_path)
    recorded: list[Entry] = []

    def overwritten(settings: Config) -> None:
        # The first edit is saved over with a typo while its backend is checked.
        if settings.llm == Claude():
            home.config.write_text('[llm]\nbackend = "claud"\n')

    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, recorded.append, overwritten, period=0.01), 0.5))
    await asyncio.sleep(0.05)
    home.config.write_text('[llm]\nbackend = "claude"\n')
    with pytest.raises(TimeoutError):
        await watching
    assert [type(entry) for entry in recorded] == [SettingsEdited] and "'claud'" in str(recorded[0])


def test_the_default_whisper_model_is_pipecats_large_v3_turbo() -> None:
    from pipecat.services.whisper.stt import MLXModel

    assert config.WHISPER_MODEL == MLXModel.LARGE_V3_TURBO
