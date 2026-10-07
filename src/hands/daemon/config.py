"""The settings file, `<home>/config.toml`: read once, when a run starts, into a frozen Config; an edit to it while the
run runs starts the run again, on the file as edited.

    [llm]
    backend = "claude"           # the brain, a Claude Code of hands' own: the default, and the one there is
    model = "claude-sonnet-5-5"  # one of CLAUDE_MODELS

    [telemetry]
    collector = "http://otel.example:4318"   # an OpenTelemetry collector's OTLP/HTTP address; none by default

    [talk]
    personality = "Dry and wry, with a bit of wit."   # how hands comes across, in the user's words; hands' own by default
    wake_word = "Hey Mycroft"     # what the wake word trigger listens for: Hey Jarvis (the default), Hey Mycroft, Hey Rhasspy, or Alexa
    wake_word_model = "~/wake/hey_computer.onnx"      # or a model of your own, trained with openWakeWord on wake_word, said as written

A file left out, or a key, is the default. Secrets are not settings: the brain's login is in its own config directory. The voice is not either: the user chooses it by voice,
and it changes while the daemon runs (hands.voice.voices). The model is a setting the user may also choose by voice:
that choice is an edit to this file (OwnModel), taken as any edit is. `hands run --model` names one that outranks the
file's for as long as that run runs, its restarts included.

[LAW:no-mode-explosion] the settings cap: every key names a variant or a value, and there are no flags. A key that
would be a flag is a variant with a real alternative, or it does not exist.
"""

import asyncio
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import tomlkit

from hands.sessions.audit import Record, SettingsEdited
from hands.sessions.files import replace_whole
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.threads import off_loop
from hands.voice.wakeword import PRETRAINED, Pretrained, Trained, Word

# How late an edit to the file is heard.
EDIT_SECONDS = 1.0

ANTHROPIC_MODEL = "claude-sonnet-5-5"
# The models hands runs on: one Haiku, Sonnet, Opus, and Fable each. A model outside them is refused where the file is
# parsed, since by voice a turn that fails on it could never choose another.
CLAUDE_MODELS = ("claude-haiku-4-5-20251001", ANTHROPIC_MODEL, "claude-opus-5-5", "claude-fable-5-1")


@dataclass(frozen=True)
class Claude:
    """Claude through a slim Claude Code of hands' own, on its own login; its requests go through hands' proxy, so it has no URL."""

    model: str = ANTHROPIC_MODEL


@dataclass(frozen=True)
class Config:
    """`collector` is the OpenTelemetry collector each wide event is also sent to, over OTLP/HTTP; None sends them nowhere
    but the audit log. `personality` is how hands comes across, in the user's own words; None is hands' own. `wake` is
    the wake word the wake word trigger listens for."""

    llm: Claude = Claude()
    collector: str | None = None
    personality: str | None = None
    wake: Word = Pretrained()


@dataclass(frozen=True)
class Settings:
    """The settings a run starts on: the file's bytes as it read them, None where there was no file, and what they hold
    with `model` in place of the file's, the one `hands run --model` named, None where the run was given none."""

    held: bytes | None
    config: Config
    model: str | None = None

    def path(self, home: Home) -> Path | None:
        """The file the settings were read from, None where they are every default."""
        return None if self.held is None else home.config


def load(home: Home, model: str | None = None) -> Settings:
    """The settings `home` holds, on `model` where one is given, read once as a run starts; raises Rejected naming the
    file and what is wrong in it, or --model and why hands cannot run on it."""
    held = _readable(home, _held(home))
    return Settings(held, _settings(home, held, model), model)


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
            # [LAW:one-source-of-truth] weighed on the run's --model, as the run's own settings were, so an edit to the
            # file's model alone is no edit while one outranks it.
            settings = _settings(home, _readable(home, now), running.model)
            if settings != running.config:
                # reachable blocks, on the brain's login check, on a thread a stop does not wait for.
                await off_loop(partial(reachable, settings), "weighing a settings edit")
                if _held(home) == now:
                    return SettingsEdited(path=str(home.config), refused=None)
                # Saved over while weighed: no verdict, so these bytes are weighed again should they come back.
                continue
        except Rejected as error:
            record(SettingsEdited(path=str(home.config), refused=str(error)))
        weighed = now


class OwnModel:
    """hands' own model: the one the run is on, and the one the user chooses for hands by voice.

    [LAW:one-source-of-truth] config.toml is the one place the model is chosen, by hand or by voice, and `hands run
    --model` the one thing that outranks it, for that run alone: a choice by voice is refused while it does. A choice by
    voice is an edit to the file, its every other line kept as written, and the run takes it as it takes any edit: by
    starting again on it once `edited` has weighed it.
    """

    def __init__(self, home: Home, running: Settings, reachable: Callable[[Config], object]) -> None:
        self._home = home
        self._running = running
        self._reachable = reachable

    def running(self) -> str:
        """The model the run is on: the one --model named, or the file's as it started."""
        return self._running.config.llm.model

    def weigh(self, model: str) -> Callable[[], None]:
        """The edit that runs hands on `model`, written by calling it. Raises Rejected where the file could not take it, or
        a start on it could not reach its model: weighed here as `edited` weighs an edit by hand, so a choice by voice is
        refused while the user is there to hear why, never written to be refused after."""
        if self._running.model is not None:
            # [LAW:one-source-of-truth] the file's model is not the run's while --model outranks it, so an edit to it would
            # change nothing until hands is started without the flag.
            raise Rejected(f"hands was started with --model {self._running.model}, which it keeps over {self._home.config} until it is started without it")
        model = model.strip()
        held = _readable(self._home, _held(self._home))
        edited = _with_model(held, model)
        chosen = _settings(self._home, edited)
        if chosen == self._running.config:
            raise Rejected(f"hands runs on {model} already")
        if chosen == _settings(self._home, held):
            # The file names it, and the run is not on it: an edit `edited` has yet to take, or one it refused, which the
            # same bytes written again would not change. Which of the two is not known here, so the reason names both.
            raise Rejected(f"{self._home.config} names {model} already and hands is not on it yet: an edit it is about to take, or one it refused when saved, which `hands log` says and `hands restart` starts on")
        self._reachable(chosen)
        return partial(_keep, self._home, held, edited)


def _with_model(held: bytes | None, model: str) -> bytes:
    # Only read once `_settings` has taken the file, so it is TOML and [llm], where there is one, is a table.
    document = tomlkit.parse(b"" if held is None else held)
    llm = cast(dict[str, object], document.setdefault("llm", tomlkit.table()))
    llm["model"] = model
    return tomlkit.dumps(document).encode()


def _keep(home: Home, held: bytes | None, edited: bytes) -> None:
    # [LAW:one-source-of-truth] one writer at a time: a save by hand since the choice was weighed is the user's, and is
    # never written over.
    if _held(home) != held:
        raise Rejected(f"{home.config} was saved after the model was chosen, so the choice was not written over it")
    # Through a link to the file, as dotfiles keep one, so the link stays and the file it names takes the edit.
    target = home.config.resolve()
    replace_whole(target, edited.decode(), 0o644 if held is None else target.stat().st_mode & 0o777)


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


def _settings(home: Home, held: bytes | None, model: str | None = None) -> Config:
    # [LAW:single-enforcer] the start and the watch weigh the file's bytes here, the same bytes each compared, on the
    # same --model.
    try:
        return parse("" if held is None else held.decode(), model)
    except UnicodeDecodeError as error:
        raise Rejected(f"{home.config} could not be read: {error}") from error
    except ModelFlagRejected:
        raise
    except Rejected as error:
        raise Rejected(f"{home.config}: {error}") from error


class ModelFlagRejected(Rejected):
    """A model --model named that hands cannot run on: the flag's fault, never the file's, so it is said without it."""


def parse(text: str, model: str | None = None) -> Config:
    """The settings `text` holds, on `model`, the one --model names, where one is given: it outranks the file's, which
    is then never read, so one hands does not offer refuses nothing and an edit to it alone is no edit."""
    # [LAW:parse-dont-validate] the one crossing from the file: past it, a setting is a variant that holds exactly its fields.
    try:
        top = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise Rejected(f"not TOML: {error}") from error
    _known(top, "the file", ("llm", "telemetry", "talk"))
    telemetry = _table(top, "telemetry")
    _known(telemetry, "[telemetry]", ("collector",))
    talk = _table(top, "talk")
    _known(talk, "[talk]", ("personality", "wake_word", "wake_word_model"))
    return Config(llm=_llm(_table(top, "llm"), model), collector=_collector(telemetry), personality=_personality(talk), wake=_wake(talk))


def _wake(table: Mapping[str, object]) -> Word:
    """The wake word: one of openWakeWord's own, matched however it is capitalised, or, with a model, the user's own,
    said as written. Whether the model is there is the switch to the wake word's to find (hands.voice.trigger.readied),
    as it is for openWakeWord's own."""
    if "wake_word" not in table:
        if "wake_word_model" in table:
            raise Rejected("[talk] wake_word_model needs wake_word, the phrase the model was trained on, which hands tells you to say")
        return Pretrained()
    said = " ".join(_text(table, "[talk]", "wake_word", "").split())
    if "wake_word_model" in table:
        model = Path(_text(table, "[talk]", "wake_word_model", "")).expanduser()
        if not model.is_absolute() or model.suffix != ".onnx":
            raise Rejected(f"[talk] wake_word_model {str(model)!r} is not a full path to an ONNX model, such as ~/wake/hey_computer.onnx")
        return Trained(phrase=said, model=model)
    for phrase in PRETRAINED:
        if said.casefold() == phrase.casefold():
            return Pretrained(phrase)
    raise Rejected(f"[talk] wake_word {said!r} is not one of openWakeWord's own: {', '.join(PRETRAINED)}; a wake word of your own takes wake_word_model too, the model you trained on it")


def _personality(table: Mapping[str, object]) -> str | None:
    return _text(table, "[talk]", "personality", "") if "personality" in table else None


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


def _llm(table: Mapping[str, object], model: str | None) -> Claude:
    # [LAW:one-type-per-behavior] hands reaches its model through a harness it drives, as it drives Claude Code: the brain.
    # Another harness joins as another backend here; a model API reached with a key is none.
    match backend := _text(table, "[llm]", "backend", "claude"):
        case "claude":
            # A url would be ignored, its requests going through hands' proxy to Anthropic's API, so it is refused.
            _known(table, "[llm] for claude", ("backend", "model"))
            return _on(Claude(), table, model)
        case _:
            raise Rejected(f"[llm] backend {backend!r} is not one hands runs on: it runs on claude, the brain, a Claude Code of its own, never on a model API reached with a key")


def _on(llm: Claude, table: Mapping[str, object], flag: str | None) -> Claude:
    """`llm` on the model `table` names, its default where it names none, or on `flag`, the --model that outranks it."""
    if flag is None:
        return replace(llm, model=_offered(_text(table, "[llm]", "model", llm.model)))
    try:
        return replace(llm, model=_offered(flag))
    except Rejected as error:
        raise ModelFlagRejected(f"--model {flag!r}: {error}") from error


def _offered(model: str) -> str:
    # [LAW:single-enforcer] the one place a model is refused that hands does not offer, whether the file, --model, or a
    # choice by voice named it.
    if model not in CLAUDE_MODELS:
        raise Rejected(f"hands runs Claude on {', '.join(CLAUDE_MODELS)}, not {model}")
    return model


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
