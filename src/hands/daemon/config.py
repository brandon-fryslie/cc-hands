"""The settings file, `<home>/config.toml`: read once, when a run starts, into a frozen Config; an edit to it while the
run runs starts the run again, on the file as edited.

    [llm]
    backend = "claude"           # "anthropic" (the default), "openai", or "claude", the brain
    model = "claude-sonnet-5"    # any backend's
    url = "https://..."          # "anthropic" and "openai" only: another server that speaks the API

    [transcription]
    url = "http://127.0.0.1:8610/v1"   # the server each hold is transcribed by: LowTalker's, or another speaking OpenAI's

    [telemetry]
    collector = "http://otel.example:4318"   # an OpenTelemetry collector's OTLP/HTTP address; none by default

A file left out, or a key, is the default. Secrets are not settings: an API key comes from the environment or the
keychain, and the brain's login from its own config directory. The voice is not either: the user chooses it by voice,
and it changes while the daemon runs (hands.voice.voices).

[LAW:no-mode-explosion] the settings cap: every key names a variant or a value, and there are no flags. A key that
would be a flag is a variant with a real alternative, or it does not exist.
"""

import asyncio
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from hands.sessions.audit import Record, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.threads import off_loop

# Where LowTalker's network build serves transcription, on loopback.
TRANSCRIPTION_URL = "http://127.0.0.1:8610/v1"
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
    """`transcription` is the base of the server each hold is uploaded to, which /audio/transcriptions is appended to.
    `collector` is the OpenTelemetry collector each wide event is also sent to, over OTLP/HTTP; None sends them nowhere
    but the audit log."""

    llm: LLM = Anthropic()
    transcription: str = TRANSCRIPTION_URL
    collector: str | None = None


@dataclass(frozen=True)
class Settings:
    """The settings a run starts on: the file's bytes as it read them, None where there was no file, and what they hold."""

    held: bytes | None
    config: Config

    def path(self, home: Home) -> Path | None:
        """The file the settings were read from, None where they are every default."""
        return None if self.held is None else home.config


def load(home: Home) -> Settings:
    """The settings `home` holds, read once as a run starts; raises Rejected naming the file and what is wrong in it."""
    held = _readable(home, _held(home))
    return Settings(held, _settings(home, held))


async def edited(home: Home, record: Record, reachable: Callable[[Config], object], running: Settings, period: float = EDIT_SECONDS) -> SettingsEdited:
    """The edit, once the file holds settings other than `running`, the ones the run was started on: written, rewritten,
    or removed.

    [LAW:no-ambient-temporal-coupling] weighed against the bytes the run read, not a read of its own, so an edit is never
    missed, however late after the run's read the watch begins. An edit is weighed once its
    bytes read the same on two polls, so a save an editor writes in two steps is weighed whole, and it is taken only
    while the file still holds it once weighed. One that does not parse, or names a model `reachable` refuses, is
    said and outlived, and the run keeps the settings it has; the next edit is weighed as any other. One whose
    settings are the run's, a comment or a revert, is no edit.
    """
    seen: bytes | _Unreadable | None = running.held
    weighed = seen
    while True:
        await asyncio.sleep(period)
        if (now := _held(home)) != seen:
            seen = now
            continue
        if now == weighed:
            continue
        try:
            settings = _settings(home, _readable(home, now))
            if settings != running.config:
                # reachable blocks, on a keychain prompt or a login check, on a thread a stop does not wait for.
                await off_loop(partial(reachable, settings), "weighing a settings edit")
                if _held(home) == now:
                    return SettingsEdited(path=str(home.config), refused=None)
                # Saved over while weighed: no verdict, so these bytes are weighed again should they come back.
                continue
        except Rejected as error:
            record(SettingsEdited(path=str(home.config), refused=str(error)))
        weighed = now


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


def _readable(home: Home, held: bytes | _Unreadable | None) -> bytes | None:
    match held:
        case _Unreadable(reason=reason):
            raise Rejected(f"{home.config} could not be read: {reason}")
        case _:
            return held


def _settings(home: Home, held: bytes | None) -> Config:
    # [LAW:single-enforcer] the start and the watch weigh the file's bytes here, the same bytes each compared.
    if held is None:
        return Config()
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
    _known(top, "the file", ("llm", "transcription", "telemetry"))
    transcription = _table(top, "transcription")
    _known(transcription, "[transcription]", ("url",))
    telemetry = _table(top, "telemetry")
    _known(telemetry, "[telemetry]", ("collector",))
    return Config(llm=_llm(_table(top, "llm")), transcription=_transcription(transcription), collector=_collector(telemetry))


def _transcription(table: Mapping[str, object]) -> str:
    return _base(_text(table, "[transcription]", "url", TRANSCRIPTION_URL), "[transcription] url", "a transcription server", TRANSCRIPTION_URL, "/audio/transcriptions")


def _collector(table: Mapping[str, object]) -> str | None:
    if "collector" not in table:
        return None
    return _base(_text(table, "[telemetry]", "collector", ""), "[telemetry] collector", "an OTLP/HTTP collector", "http://host:4318", "/v1/traces")


def _base(url: str, where: str, server: str, example: str, appended: str) -> str:
    """[LAW:parse-dont-validate] a server's base address, spelled one way from here on: the base `appended` is appended
    to, with no trailing slash."""
    url = url.rstrip("/")
    # Not echoed: a refused edit is a log line, and the address may hold credentials.
    unusable = Rejected(f"{where} is not {server}'s base address, as {example}, to which hands appends {appended}, with no credentials in it")
    try:
        parts = urlsplit(url)
        # Read for what it raises: a port that is not a number, or out of range.
        parts.port
    except ValueError as error:
        raise unusable from error
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment or parts.path.endswith(appended):
        raise unusable
    return url


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
