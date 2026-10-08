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

from hands.sessions.payload import Payload, Rejected


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


def unanswered(state: Path, cwd: Path, key: Key | None) -> Unanswered | None:
    """Whether a Claude Code with the .claude.json `state`, started in `cwd` with `key`, would open on one of its first-run
    questions; None when it would not. `cwd` is trusted when it or a directory above it is."""
    onboarding = ("its onboarding unfinished", "a theme and a login")
    trust = (f"{cwd} untrusted", f"whether to trust {cwd}")
    use = [] if key is None else [(f"the API key {key.source} sets unanswered", f"whether to use the API key {key.source} sets")]
    try:
        said = Payload.parse(state.read_bytes())
        projects = Payload.of(said.fields.get("projects", {}), "its projects").fields
        trusted = any(Payload.of(projects[place], place).optional_flag("hasTrustDialogAccepted") for place in map(str, (cwd, *cwd.parents)) if place in projects)
        responses = Payload.of(said.fields.get("customApiKeyResponses", {}), "its API key answers")
        answered = {*responses.optional_items("approved"), *responses.optional_items("rejected")}
        finished = said.optional_flag("hasCompletedOnboarding")
    except FileNotFoundError:
        return Unanswered(f"no {state}", tuple(asks for _, asks in (onboarding, trust, *use)))
    except Rejected as error:
        # Claude Code's own first run is what writes this state, so it is the run that mends one hands cannot read.
        return Unanswered(f"{state} unreadable: {error}", tuple(asks for _, asks in (onboarding, trust, *use)))
    # [LAW:dataflow-not-control-flow] each question is open or not by its own record; the run asks every open one.
    open_ = [question for question, done in ((onboarding, finished), (trust, trusted), *((question, key.value[-20:] in answered) for question in use if key)) if not done]
    if not open_:
        return None
    return Unanswered("; ".join(why for why, _ in open_), tuple(asks for _, asks in open_))


def settings_key(settings: Path) -> Key | None:
    """The API key `settings` puts in Claude Code's environment, over any the process has; raises Rejected when it cannot be read."""
    try:
        raw = settings.read_bytes()
    except FileNotFoundError:
        return None
    value = Payload.of(Payload.parse(raw).fields.get("env", {}), "its env").optional_text("ANTHROPIC_API_KEY")
    return None if value is None else Key(value, str(settings))


def state_of(environment: Mapping[str, str]) -> Path:
    """The .claude.json a Claude Code with this environment keeps its first-run answers in: in CLAUDE_CONFIG_DIR when it is
    set, else in the home, beside .claude, not in it (2.1.289)."""
    return Path(environment.get("CLAUDE_CONFIG_DIR") or environment.get("HOME") or Path.home()) / ".claude.json"


def persons(environment: Mapping[str, str], cwd: Path, config_dir: Path) -> Unanswered | None:
    """What the person's own Claude Code, run with this environment under `config_dir`, would ask first in `cwd`; raises
    Rejected when its settings.json cannot be read, which no first run mends."""
    environmental = environment.get("ANTHROPIC_API_KEY")
    key = settings_key(config_dir / "settings.json") or (None if environmental is None else Key(environmental, "ANTHROPIC_API_KEY in your environment"))
    return unanswered(state_of(environment), cwd.resolve(), key)
