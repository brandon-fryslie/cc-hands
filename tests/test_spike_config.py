"""The process boundary: which LLM backend the environment names."""

import pytest

from hands.spike import LOCAL_LLM_MODEL, LOCAL_LLM_URL, backend_from_env
from hands.voice.pipeline import AnthropicBackend, OpenAICompatibleBackend


def test_default_is_the_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HANDS_LLM", "HANDS_LLM_URL", "HANDS_LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert backend_from_env() == OpenAICompatibleBackend(base_url=LOCAL_LLM_URL, model=LOCAL_LLM_MODEL)


def test_local_url_and_model_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "local")
    monkeypatch.setenv("HANDS_LLM_URL", "http://elsewhere:9/v1")
    monkeypatch.setenv("HANDS_LLM_MODEL", "some/model")
    assert backend_from_env() == OpenAICompatibleBackend(base_url="http://elsewhere:9/v1", model="some/model")


def test_anthropic_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        backend_from_env()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("HANDS_LLM_MODEL", raising=False)
    assert backend_from_env() == AnthropicBackend(api_key="k", model="claude-haiku-4-5-20251001")


def test_unknown_choice_stops_at_the_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDS_LLM", "gpt")
    with pytest.raises(SystemExit):
        backend_from_env()
