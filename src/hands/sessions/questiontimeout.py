"""How long a session's Claude Code waits on a question before it continues without an answer: its askUserQuestionTimeout.

Claude Code puts the setting nowhere hands can read it whole, so it is read from the settings a person sets it in, as
2.1.289 reads them: the first of `--settings` and the user's settings.json that sets it to one of its values, the user's
only where `--setting-sources` leaves them read. Project and local settings never set it. Managed policy outranks both,
through server, MDM, file and helper sources; hands, run by the person whose session it is, reads none of them.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hands.core.session import Membership
from hands.sessions.membership import config_of
from hands.sessions.payload import Rejected
from hands.sessions.terminals import Launch, Undescribed, launch_of

# Claude Code's values for the setting, in seconds; "never" waits as long as it takes, as an unset one does (2.1.289).
_SECONDS: Mapping[str, float | None] = {"60s": 60.0, "5m": 300.0, "10m": 600.0, "never": None}

# The settings it is read from, named as Claude Code names their sources, in the order the first that sets it wins.
Tier = Literal["flagSettings", "userSettings"]


@dataclass(frozen=True)
class Unread:
    """Settings that could not be read, and why."""

    why: str


# The settings a tier holds: None where it holds none, as a session started with no --settings, or no settings.json.
Settings = Mapping[str, object] | Unread | None


@dataclass(frozen=True)
class QuestionTimeout:
    """The session's question timeout: seconds, or None where it waits on a question as long as it takes; the settings
    that set it, None where none did; and each that could not be read, and why."""

    seconds: float | None
    set_by: Tier | None
    unread: tuple[str, ...]


def question_timeout(membership: Membership) -> QuestionTimeout:
    """The question timeout the session's own Claude Code runs on, from its settings as they read now."""
    return timeout_of(_launch(membership.pid), _config(membership))


def timeout_of(launch: Launch | Unread, config: Path | Unread) -> QuestionTimeout:
    """The question timeout of a claude started so, under that config directory."""
    tiers: list[tuple[Tier, Settings]] = [("flagSettings", _flag_settings(launch)), ("userSettings", _user_settings(launch, config))]
    unread = tuple(f"{tier}: {settings.why}" for tier, settings in tiers if isinstance(settings, Unread))
    given: list[tuple[Tier, str]] = [(tier, value) for tier, settings in tiers if (value := _value(settings)) is not None]
    match given:
        case [(tier, value), *_]:
            return QuestionTimeout(_SECONDS[value], tier, unread)
        case _:
            return QuestionTimeout(None, None, unread)


def _value(settings: Settings) -> str | None:
    """What the settings set it to, of the values Claude Code has: any other is unset, as Claude Code drops it."""
    match settings:
        case Mapping():
            value = settings.get("askUserQuestionTimeout")
            return value if isinstance(value, str) and value in _SECONDS else None
        case Unread() | None:
            return None


def _flag_settings(launch: Launch | Unread) -> Settings:
    match launch:
        case Unread():
            return launch
        case Launch(directory=directory, arguments=arguments):
            match _option(arguments, "--settings"):
                case None:
                    return None
                case named if named.strip().startswith("{") and named.strip().endswith("}"):
                    # JSON in place, as Claude Code takes a value in braces.
                    return _settings(named)
                case named:
                    # Claude Code will not start without the file it names, so one missing was looked for in the wrong place.
                    return _read(directory / named, missing=Unread(f"{directory / named}: not found"))


def _user_settings(launch: Launch | Unread, config: Path | Unread) -> Settings:
    match launch, config:
        case _, Unread():
            return config
        case Launch(arguments=arguments), Path() if not _reads_user(_option(arguments, "--setting-sources")):
            return None
        case _, Path():
            return _read(config / "settings.json", missing=None)


def _reads_user(sources: str | None) -> bool:
    """Whether a claude started with these `--setting-sources` reads the user's settings: every source, given none."""
    return sources is None or "user" in (source.strip() for source in sources.split(","))


def _option(arguments: Sequence[str], name: str) -> str | None:
    """What the last of an option among a claude's arguments gives, as the last of a repeated option wins."""
    given = None
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            given = arguments[index + 1]
        elif argument.startswith(f"{name}="):
            given = argument.removeprefix(f"{name}=")
    return given


def _launch(pid: int) -> Launch | Unread:
    try:
        launch = launch_of(pid)
    except OSError as error:
        return Unread(str(error))
    match launch:
        case Undescribed(call=call, errno=errno):
            return Unread(f"the kernel refused {call} (errno {errno})")
        case None:
            return Unread("the session's process has exited")
        case Launch():
            return launch


def _config(membership: Membership) -> Path | Unread:
    try:
        return config_of(membership)
    except Rejected as error:
        return Unread(str(error))


def _read(path: Path, *, missing: Settings) -> Settings:
    """The settings in a file, or `missing` where there is none."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return missing
    except OSError as error:
        return Unread(f"{path}: {error.strerror}")
    match _settings(text):
        case Unread(why=why):
            return Unread(f"{path}: {why}")
        case settings:
            return settings


def _settings(text: str) -> Mapping[str, object] | Unread:
    try:
        read = json.loads(text)
    except json.JSONDecodeError as error:
        return Unread(f"not JSON ({error.msg})")
    return cast(Mapping[str, object], read) if isinstance(read, dict) else Unread("not a JSON object")
