"""The process boundary: which LLM backend the environment names."""

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from hands.daemon import run
from hands.daemon.run import ANTHROPIC_MODEL, ANTHROPIC_URL, OPENAI_MODEL, OPENAI_URL, backend_from_env
from hands.sessions import heartbeat
from hands.sessions.audit import Entry, LLMChosen, encoded
from hands.sessions.home import Home
from hands.sessions.registry import Sessions
from hands.sessions.proxy import UPSTREAM
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, OpenAICompatibleBackend

HOME = Home(Path("/Users/someone/.hands"))


def test_default_is_claude_on_the_anthropic_api(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HANDS_LLM", "HANDS_LLM_URL", "HANDS_LLM_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert backend_from_env(HOME) == AnthropicBackend(base_url="https://api.anthropic.com", api_key="k", model=ANTHROPIC_MODEL)


def test_anthropic_url_and_model_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HANDS_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("HANDS_LLM_URL", "https://api-chicago.codexapi.pro")
    monkeypatch.setenv("HANDS_LLM_MODEL", "claude-other")
    assert backend_from_env(HOME) == AnthropicBackend(base_url="https://api-chicago.codexapi.pro", api_key="k", model="claude-other")


def test_claude_is_the_brain_on_the_login_in_hands_own_config_dir_with_no_key(monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    for var in ("HANDS_LLM_URL", "HANDS_LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HANDS_LLM", "claude")
    assert backend_from_env(HOME) == ClaudeCodeBackend(base_url=UPSTREAM, model=ANTHROPIC_MODEL, config_dir=Path("/Users/someone/.hands/brain"))
    monkeypatch.setenv("HANDS_LLM_MODEL", "claude-other")
    assert backend_from_env(HOME).model == "claude-other"


def test_a_brain_with_no_login_stops_the_run_before_the_voice_loads_naming_the_command(monkeypatch: pytest.MonkeyPatch, fake_claude: Path) -> None:
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.setenv("HANDS_LLM", "claude")
    monkeypatch.setenv("LOGGED_IN", "0")
    with pytest.raises(SystemExit, match="CLAUDE_CONFIG_DIR=/Users/someone/.hands/brain claude auth login"):
        backend_from_env(HOME)


def test_a_url_for_the_brain_is_refused_rather_than_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "claude")
    monkeypatch.setenv("HANDS_LLM_URL", "http://localhost:8080")
    with pytest.raises(SystemExit, match="does not apply to HANDS_LLM=claude"):
        backend_from_env(HOME)


def test_the_keychain_key_never_leaves_for_another_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HANDS_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HANDS_LLM_URL", "https://api-chicago.codexapi.pro")
    monkeypatch.setattr(run, "keychain_password", {"HANDS_LLM_ANT_KEY": "anthropic-own"}.get)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        backend_from_env(HOME)


def test_a_blank_url_is_no_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    monkeypatch.setenv("HANDS_LLM_URL", "  ")
    monkeypatch.setattr(run, "keychain_password", {"HANDS_LLM_ANT_KEY": "anthropic-own"}.get)
    monkeypatch.setenv("HANDS_LLM", "anthropic")
    assert backend_from_env(HOME) == AnthropicBackend(base_url=ANTHROPIC_URL, api_key="anthropic-own", model=ANTHROPIC_MODEL)
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert backend_from_env(HOME) == OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="k", model=OPENAI_MODEL)


def test_an_anthropic_url_with_its_own_v1_stops_at_the_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HANDS_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    for url in ("http://localhost:8080/v1", "https://api-chicago.codexapi.pro/v1/"):
        monkeypatch.setenv("HANDS_LLM_URL", url)
        with pytest.raises(SystemExit, match="ends in /v1"):
            backend_from_env(HOME)


def test_the_local_variant_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "local")
    with pytest.raises(SystemExit, match="HANDS_LLM='local' is not one of: anthropic, openai"):
        backend_from_env(HOME)


def test_anthropic_needs_its_key_from_the_environment_or_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    kept: dict[str, str] = {}
    monkeypatch.setattr(run, "keychain_password", kept.get)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set and the keychain holds no HANDS_LLM_ANT_KEY"):
        backend_from_env(HOME)
    kept["HANDS_LLM_ANT_KEY"] = "from-keychain"
    assert backend_from_env(HOME) == AnthropicBackend(base_url=ANTHROPIC_URL, api_key="from-keychain", model=ANTHROPIC_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert backend_from_env(HOME) == AnthropicBackend(base_url=ANTHROPIC_URL, api_key="k", model=ANTHROPIC_MODEL)


def test_openai_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env(HOME)
    # An empty line in a .env reads as set to nothing, which is no key either.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env(HOME)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env(HOME)
    # Space around a key in a .env line is not part of the key.
    monkeypatch.setenv("OPENAI_API_KEY", " k ")
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    assert backend_from_env(HOME) == OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="k", model=OPENAI_MODEL)


def test_openai_reaches_openai_with_its_key_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    assert backend_from_env(HOME) == OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key="k", model=OPENAI_MODEL)


def test_openai_url_and_model_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("HANDS_LLM_URL", "https://reseller.example/v1")
    monkeypatch.setenv("HANDS_LLM_MODEL", "gpt-other")
    assert backend_from_env(HOME) == OpenAICompatibleBackend(base_url="https://reseller.example/v1", api_key="k", model="gpt-other")


def test_a_backend_printed_does_not_print_its_key() -> None:
    """The eval prints the backend it runs on, and a key printed is a key leaked."""
    for backend in (
        OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="sk-secret", model=OPENAI_MODEL),
        AnthropicBackend(base_url=ANTHROPIC_URL, api_key="sk-secret", model=ANTHROPIC_MODEL),
    ):
        assert "sk-secret" not in repr(backend) and "sk-secret" not in str(backend)


def test_unknown_choice_stops_at_the_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "gpt")
    with pytest.raises(SystemExit):
        backend_from_env(HOME)


def _starting(tmp_path: Path) -> tuple[Home, Sessions, heartbeat.Heart, run.VoiceConfig]:
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.01))
    sessions = Sessions(permission_deadline=60.0, clock=time.monotonic, record=lambda _event: None)
    config = run.VoiceConfig(llm=AnthropicBackend(base_url=ANTHROPIC_URL, api_key="sk-secret", model=ANTHROPIC_MODEL), whisper_model="w", voice="v")
    return Home(tmp_path), sessions, heart, config


async def test_the_start_beats_while_the_configuration_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A keychain prompt answered slowly is a start that is waiting, never one that looks stuck.
    home, sessions, heart, config = _starting(tmp_path)
    voice = cast(run.Voice, object())

    def built(_config: run.VoiceConfig, tools: object) -> run.Voice:
        return voice

    monkeypatch.setattr(run, "build_voice", built)
    answered = threading.Event()

    def prompted() -> run.VoiceConfig:
        answered.wait()
        return config

    recorded: list[Entry] = []
    starting = asyncio.create_task(run.start(prompted, lambda: None, home, sessions, [], heart, asyncio.Event(), recorded.append))
    # The prompt is answered only once the start has said "starting" three times while it waited.
    beats: set[datetime] = set()
    while len(beats) < 3:
        status = heartbeat.read(heart.path)
        if status is not None and status.pipeline == "starting":
            beats.add(status.written_at)
        await asyncio.sleep(0.005)
    answered.set()
    assert await starting == (config, voice)
    # The log says which server and model the run reaches, and never with what key.
    assert recorded == [LLMChosen(backend="AnthropicBackend", base_url=ANTHROPIC_URL, model=ANTHROPIC_MODEL)]
    assert "sk-secret" not in str([encoded(entry) for entry in recorded])


async def test_a_stop_during_the_configuration_read_ends_the_start(tmp_path: Path) -> None:
    home, sessions, heart, _config = _starting(tmp_path)
    never = threading.Event()

    def prompted() -> run.VoiceConfig:
        never.wait()
        raise AssertionError("the prompt was never answered")

    quit_event = asyncio.Event()
    quit_event.set()
    assert await run.start(prompted, lambda: None, home, sessions, [], heart, quit_event, lambda _event: None) is None
    never.set()


def test_a_refused_configuration_stops_the_start(tmp_path: Path) -> None:
    # Run as the CLI runs it: a SystemExit from a task leaves the event loop itself, so only asyncio.run's caller sees it.
    home, sessions, heart, _config = _starting(tmp_path)

    def refused() -> run.VoiceConfig:
        raise SystemExit("no key")

    with pytest.raises(SystemExit, match="no key"):
        asyncio.run(run.start(refused, lambda: None, home, sessions, [], heart, asyncio.Event(), lambda _event: None))
