"""The settings file, `<home>/config.toml`: read once, when a run starts, into a frozen Config; an edit to it while the
run runs starts the run again, on the file as edited.

    [llm]
    backend = "claude"           # "anthropic" (the default), "openai", or "claude", the brain
    model = "claude-sonnet-5"    # any backend's
    url = "https://..."          # "anthropic" and "openai" only: another server that speaks the API

    [whisper]
    model = "mlx-community/whisper-large-v3-turbo"

A file left out, or a key, is the default. Secrets are not settings: an API key comes from the environment or the
keychain, and the brain's login from its own config directory. The voice is not either: the user chooses it by voice,
and it changes while the daemon runs (hands.voice.voices).

[LAW:no-mode-explosion] the settings cap: every key names a variant or a value, and there are no flags. A key that
would be a flag is a variant with a real alternative, or it does not exist.
"""

import asyncio
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hands.sessions.audit import Record, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.payload import Rejected

# Spelled here, not taken from Pipecat's Whisper service, whose import is most of the seconds the start spends off
# the loop: the start watches this file before then, on it.
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
# How late an edit to the file is heard.
EDIT_SECONDS = 1.0

# The SDK appends /v1/messages to this, so an Anthropic-compatible server's URL has no /v1 of its own.
ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_URL = "https://api.openai.com/v1"
# Not a reasoning model, so no thinking precedes the first spoken word; it calls tools and takes max_tokens.
OPENAI_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True)
class Anthropic:
    """Claude over the Anthropic API, or another server that speaks it."""

    url: str = ANTHROPIC_URL
    model: str = ANTHROPIC_MODEL


@dataclass(frozen=True)
class OpenAI:
    """An OpenAI chat completions server: OpenAI's own, or another that speaks it."""

    url: str = OPENAI_URL
    model: str = OPENAI_MODEL


@dataclass(frozen=True)
class Claude:
    """Claude through a slim Claude Code of hands' own, on the subscription; its requests go through hands' proxy, so it has no URL."""

    model: str = ANTHROPIC_MODEL


type LLM = Anthropic | OpenAI | Claude


@dataclass(frozen=True)
class Config:
    llm: LLM = Anthropic()
    whisper_model: str = WHISPER_MODEL


def load(home: Home) -> tuple[Config, Path | None]:
    """The settings `home` holds and the file they were read from, or the defaults and None where it holds no file;
    raises Rejected naming the file and what is wrong in it."""
    held = _held(home)
    return _settings(home, held), None if held is None else home.config


async def edited(home: Home, record: Record, period: float = EDIT_SECONDS) -> SettingsEdited:
    """The edit, once the file holds settings other than it held when this began: written, rewritten, or removed.

    [LAW:no-ambient-temporal-coupling] begun before the run reads the file, so an edit is never missed: one that lands
    between the two starts a run already on it again, which is the same run once more. An edit is weighed once its
    bytes read the same on two polls, so a save an editor writes in two steps is weighed whole. One that does not
    parse is said and outlived, and the run keeps the settings it has; the next edit is weighed as any other, and one
    back to the file the run is on is no edit.
    """
    on = seen = weighed = _held(home)
    while True:
        await asyncio.sleep(period)
        if (now := _held(home)) != seen:
            seen = now
            continue
        if now in (weighed, on):
            weighed = now
            continue
        weighed = now
        try:
            _settings(home, now)
        except Rejected as error:
            record(SettingsEdited(path=str(home.config), refused=str(error)))
            continue
        return SettingsEdited(path=str(home.config), refused=None)


@dataclass(frozen=True)
class _Unreadable:
    reason: str


def _held(home: Home) -> bytes | _Unreadable | None:
    # The bytes, not the mtime: a save that changes nothing, or a touch, is no edit.
    try:
        return home.config.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        return _Unreadable(str(error))


def _settings(home: Home, held: bytes | _Unreadable | None) -> Config:
    # [LAW:single-enforcer] the start and the watch weigh the file's bytes here, the same bytes each compared.
    match held:
        case None:
            return Config()
        case _Unreadable(reason=reason):
            raise Rejected(f"{home.config} could not be read: {reason}")
        case bytes():
            try:
                return parse(held.decode())
            except UnicodeDecodeError as error:
                raise Rejected(f"{home.config} could not be read: {error}") from error
            except Rejected as error:
                raise Rejected(f"{home.config}: {error}") from error


def parse(text: str) -> Config:
    # [LAW:parse-dont-validate] the one crossing from the file: past it, a setting is a variant that holds exactly its fields.
    try:
        top = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise Rejected(f"not TOML: {error}") from error
    _known(top, "the file", ("llm", "whisper"))
    whisper = _table(top, "whisper")
    _known(whisper, "[whisper]", ("model",))
    return Config(llm=_llm(_table(top, "llm")), whisper_model=_text(whisper, "[whisper]", "model", Config.whisper_model))


def _llm(table: Mapping[str, object]) -> LLM:
    match backend := _text(table, "[llm]", "backend", "anthropic"):
        case "anthropic":
            _known(table, "[llm] for anthropic", ("backend", "model", "url"))
            # [LAW:parse-dont-validate] spelled one way from here on, so Anthropic's own server is known by equality.
            url = _text(table, "[llm]", "url", ANTHROPIC_URL).rstrip("/")
            if url.endswith("/v1"):
                raise Rejected(f"[llm] url {url!r} ends in /v1, and the Anthropic client appends /v1/messages itself; drop the /v1")
            return Anthropic(url=url, model=_text(table, "[llm]", "model", ANTHROPIC_MODEL))
        case "openai":
            _known(table, "[llm] for openai", ("backend", "model", "url"))
            return OpenAI(url=_text(table, "[llm]", "url", OPENAI_URL), model=_text(table, "[llm]", "model", OPENAI_MODEL))
        case "claude":
            # A url would be ignored, its requests going through hands' proxy to Anthropic's API, so it is refused.
            _known(table, "[llm] for claude", ("backend", "model"))
            return Claude(model=_text(table, "[llm]", "model", ANTHROPIC_MODEL))
        case _:
            raise Rejected(f"[llm] backend {backend!r} is not one of: anthropic, openai, claude")


def _table(top: Mapping[str, object], name: str) -> Mapping[str, object]:
    match value := top.get(name, {}):
        case dict():
            return cast(dict[str, object], value)
        case _:
            raise Rejected(f"{name} should be a table, got {type(value).__name__}")


def _known(table: Mapping[str, object], where: str, keys: tuple[str, ...]) -> None:
    # [LAW:no-silent-failure] a key misspelled, or one its variant has no use for, would be a setting silently not applied.
    if unknown := sorted(set(table) - set(keys)):
        raise Rejected(f"{where} has no {', '.join(map(repr, unknown))}; it takes {', '.join(keys)}")


def _text(table: Mapping[str, object], where: str, key: str, default: str) -> str:
    match value := table.get(key, default):
        # A blank value is no value: left in, it would be a server or a model named "".
        case str() if value.strip():
            return value.strip()
        case _:
            raise Rejected(f"{where} {key} should be a non-empty string, got {value!r}")
