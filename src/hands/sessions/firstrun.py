"""Claude Code's first run: the questions it asks once for a config and a directory, read from where it records the answers.

Claude Code records in its .claude.json that its onboarding finished (the theme, the login, its notes), each directory it
was told to trust, and its answer on an API key, by the key's last 20 characters (2.1.289). A Claude Code started where
one is unanswered opens on that question, not on its input, so nothing typed into it is taken until a person answers.
The brain's first run is `hands login`'s; the person's own, in the folder `hands smoke` runs its session in, is
`hands first-run`'s.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hands.sessions.claudecode import ClaudeCode
from hands.sessions.payload import Payload, Rejected

API_KEY = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class Key:
    """An API key Claude Code would use, and what sets it."""

    value: str
    source: str


@dataclass(frozen=True)
class Unanswered:
    """What Claude Code's state says is unanswered, or why it could not be read, and what its first run asks, in order."""

    why: str
    asks: tuple[str, ...]

    @property
    def listed(self) -> str:
        """What it asks, as one phrase: each question is a phrase of its own, some with an "and" inside."""
        return "; ".join(self.asks)


@dataclass(frozen=True)
class Recorded:
    """What a .claude.json records of Claude Code's first run: the directories it was told to trust, and the API keys it
    was told to use and not to, by their last 20 characters."""

    finished: bool
    trusted: frozenset[str]
    approved: frozenset[str]
    rejected: frozenset[str]


@dataclass(frozen=True)
class Blank:
    """A .claude.json that records nothing of a first run: there is none, or one Claude Code's first run writes over."""

    why: str


def recorded(claude_code: ClaudeCode, state: Path) -> Recorded | Blank:
    """What the .claude.json `state` records; raises Rejected when it is there but cannot be read, which no first run mends."""
    return recorded_in(claude_code.read(state), state)


def recorded_in(raw: bytes | None, state: Path) -> Recorded | Blank:
    """What `raw`, the bytes of the .claude.json `state` or None when there is none, records."""
    if raw is None:
        return Blank(f"no {state}")
    try:
        said = Payload.parse(raw)
        projects = Payload.of(said.fields.get("projects", {}), "its projects").fields
        # One entry per directory Claude Code was started in: one this cannot read is no trust, which the first run there
        # writes again, and leaves every other entry readable.
        trusted = frozenset(place for place, entry in projects.items() if isinstance(entry, dict) and cast(dict[str, object], entry).get("hasTrustDialogAccepted") is True)
        responses = Payload.of(said.fields.get("customApiKeyResponses", {}), "its API key answers")
        return Recorded(said.optional_flag("hasCompletedOnboarding"), trusted, frozenset(map(str, responses.optional_items("approved"))), frozenset(map(str, responses.optional_items("rejected"))))
    except Rejected as error:
        # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot parse.
        return Blank(f"{state} unreadable: {error}")


def unanswered(answers: Recorded | Blank, cwd: Path, key: Key | None) -> Unanswered | None:
    """Whether a Claude Code with these answers, started in `cwd` with `key`, would open on one of its first-run questions;
    None when it would not. `cwd` is trusted when it or a directory above it is."""
    onboarding = ("its onboarding unfinished", "a theme and a login")
    trust = (f"{cwd} untrusted", f"whether to trust {cwd}")
    use = [] if key is None else [(f"the API key {key.source} sets unanswered", f"whether to use the API key {key.source} sets")]
    match answers:
        case Blank(why=why):
            return Unanswered(why, tuple(asks for _, asks in (onboarding, trust, *use)))
        case Recorded(finished=finished, trusted=trusted_places, approved=approved, rejected=rejected):
            trusted = any(str(place) in trusted_places for place in (cwd, *cwd.parents))
            # [LAW:dataflow-not-control-flow] each question is open or not by its own record; the run asks every open one.
            open_ = [question for question, done in ((onboarding, finished), (trust, trusted), *((question, key.value[-20:] in approved | rejected) for question in use if key)) if not done]
            if not open_:
                return None
            return Unanswered("; ".join(why for why, _ in open_), tuple(asks for _, asks in open_))


def approved(answers: Recorded | Blank, key: Key | None) -> bool:
    """Whether Claude Code would use `key`: its first run was told to."""
    return key is not None and isinstance(answers, Recorded) and key.value[-20:] in answers.approved


def api_key(claude_code: ClaudeCode, settings: Path, environment: Mapping[str, str]) -> Key | None:
    """The API key Claude Code would use: the one `settings` puts in its environment, over any `environment` has, where an
    empty one is none (2.1.289). Raises Rejected when `settings` cannot be read."""
    raw = claude_code.read(settings)
    try:
        written = None if raw is None else Payload.of(Payload.parse(raw).fields.get("env", {}), "its env").optional_text(API_KEY)
    except Rejected as error:
        raise Rejected(f"{settings} unreadable: {error}") from error
    value, source = (environment.get(API_KEY), f"{API_KEY} in your environment") if written is None else (written, str(settings))
    return Key(value, source) if value else None


def state_of(environment: Mapping[str, str], config_dir: Path) -> Path:
    """The .claude.json a Claude Code with this environment, under `config_dir`, keeps its first-run answers in: in that
    directory when CLAUDE_CONFIG_DIR names it, else in the home, beside .claude, not in it (2.1.289)."""
    return (config_dir if environment.get("CLAUDE_CONFIG_DIR") else Path(environment.get("HOME") or Path.home())) / ".claude.json"


@dataclass(frozen=True)
class Persons:
    """What the person's own Claude Code would ask first, and whether it has an API key it was told to use, which is a
    login of its own."""

    unanswered: Unanswered | None
    keyed: bool


def persons(claude_code: ClaudeCode, environment: Mapping[str, str], cwd: Path, config_dir: Path) -> Persons:
    """What the person's own Claude Code, run with this environment under `config_dir`, would ask first in `cwd`; raises
    Rejected when its settings.json or .claude.json is there but cannot be read, which no first run mends."""
    answers = recorded(claude_code, state_of(environment, config_dir))
    key = api_key(claude_code, config_dir / "settings.json", environment)
    return Persons(unanswered(answers, cwd.resolve(), key), approved(answers, key))

