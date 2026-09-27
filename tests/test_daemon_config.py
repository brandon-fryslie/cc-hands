"""The process boundary: which LLM backend the environment names."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hands.daemon import run
from hands.daemon.run import ANTHROPIC_MODEL, LOCAL_LLM_KEY, LOCAL_LLM_MODEL, LOCAL_LLM_URL, OPENAI_MODEL, OPENAI_URL, backend_from_env
from hands.sessions import heartbeat
from hands.voice.pipeline import AnthropicBackend, OpenAICompatibleBackend


def test_default_is_the_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HANDS_LLM", "HANDS_LLM_URL", "HANDS_LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert backend_from_env() == OpenAICompatibleBackend(base_url=LOCAL_LLM_URL, api_key=LOCAL_LLM_KEY, model=LOCAL_LLM_MODEL)


def test_local_url_and_model_come_from_the_environment_and_no_real_key_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "local")
    monkeypatch.setenv("HANDS_LLM_URL", "http://elsewhere:9/v1")
    monkeypatch.setenv("HANDS_LLM_MODEL", "some/model")
    # An OpenAI key in the environment is not handed to the local server: it gets the placeholder it always got.
    monkeypatch.setenv("OPENAI_API_KEY", "real")
    assert backend_from_env() == OpenAICompatibleBackend(base_url="http://elsewhere:9/v1", api_key=LOCAL_LLM_KEY, model="some/model")


def test_anthropic_needs_its_key_from_the_environment_or_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    kept: dict[str, str] = {}
    monkeypatch.setattr(run, "keychain_password", kept.get)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set and the keychain holds no HANDS_LLM_ANT_KEY"):
        backend_from_env()
    kept["HANDS_LLM_ANT_KEY"] = "from-keychain"
    assert backend_from_env() == AnthropicBackend(api_key="from-keychain", model=ANTHROPIC_MODEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert backend_from_env() == AnthropicBackend(api_key="k", model=ANTHROPIC_MODEL)


def test_openai_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env()
    # An empty line in a .env reads as set to nothing, which is no key either.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        backend_from_env()
    # Space around a key in a .env line is not part of the key.
    monkeypatch.setenv("OPENAI_API_KEY", " k ")
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    assert backend_from_env() == OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="k", model=OPENAI_MODEL)


def test_openai_reaches_openai_with_its_key_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("HANDS_LLM_URL", raising=False)
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    assert backend_from_env() == OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key="k", model=OPENAI_MODEL)


def test_openai_url_and_model_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("HANDS_LLM_URL", "https://reseller.example/v1")
    monkeypatch.setenv("HANDS_LLM_MODEL", "gpt-other")
    assert backend_from_env() == OpenAICompatibleBackend(base_url="https://reseller.example/v1", api_key="k", model="gpt-other")


def test_a_backend_printed_does_not_print_its_key() -> None:
    """The eval prints the backend it runs on, and a key printed is a key leaked."""
    for backend in (
        OpenAICompatibleBackend(base_url=OPENAI_URL, api_key="sk-secret", model=OPENAI_MODEL),
        AnthropicBackend(api_key="sk-secret", model=ANTHROPIC_MODEL),
    ):
        assert "sk-secret" not in repr(backend) and "sk-secret" not in str(backend)


def test_unknown_choice_stops_at_the_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "gpt")
    with pytest.raises(SystemExit):
        backend_from_env()


async def test_the_start_beats_while_the_configuration_is_read(tmp_path: Path) -> None:
    # A keychain prompt answered slowly is a start that is waiting, never one that looks stuck.
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.05))
    config = run.VoiceConfig(llm=AnthropicBackend(api_key="k", model=ANTHROPIC_MODEL), whisper_model="w", voice="v")
    asked = datetime.now(UTC)

    def slow() -> run.VoiceConfig:
        time.sleep(0.3)
        return config

    assert await run._configured(slow, heart) == config  # pyright: ignore[reportPrivateUsage]
    status = heartbeat.read(heart.path)
    assert status is not None and status.pipeline == "starting"
    assert status.written_at - asked >= timedelta(seconds=0.2)


def test_a_refused_configuration_stops_the_start(tmp_path: Path) -> None:
    # Run as the CLI runs it: a SystemExit from a task leaves the event loop itself, so only asyncio.run's caller sees it.
    heart = heartbeat.Heart(tmp_path / "status.json", pid=4242, started_at=datetime.now(UTC), period=timedelta(seconds=0.05))

    def refused() -> run.VoiceConfig:
        raise SystemExit("no key")

    with pytest.raises(SystemExit, match="no key"):
        asyncio.run(run._configured(refused, heart))  # pyright: ignore[reportPrivateUsage]
