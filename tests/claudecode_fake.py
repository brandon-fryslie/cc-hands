"""The fake of `hands.sessions.claudecode.ClaudeCode` (docs/testing.md): Claude Code as the recordings in
fixtures/claudecode show it, and a person at its terminal who answers what it asks as a test chooses.

Every answer it gives is a recording: the bytes and exit a real Claude Code gave, chosen by the state a test sets, the
logins, plugins, and marketplaces each config directory holds, and its files, which live here in memory. What hands
asked and what the person was shown land in `transcript`, in order, beside what hands told the person first (`told`).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from hands.sessions import firstrun
from hands.sessions.claudecode import Answer, Instance, Method, Unreachable
from hands.sessions.payload import Rejected

RECORDINGS = Path(__file__).parent / "fixtures" / "claudecode"

type Login = Literal["claude.ai", "console"]
# The credentials Claude Code reads from its environment, which settings.json's env is put over.
TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
type Source = Literal["github", "checkout"]


def recording(name: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads((RECORDINGS / f"{name}.json").read_text()))


def answered(name: str) -> Answer:
    """A recorded subcommand's answer."""
    said = recording(name)
    return Answer(cast(int, said["exit"]), cast(str, said["stdout"]).encode(), cast(str, said["stderr"]).encode())


FIRST_RUN = recording("first-run")
SCREENS = cast(dict[str, str], FIRST_RUN["screens"])
STATES = cast(dict[str, dict[str, object]], FIRST_RUN["states"])
INSTALL = recording("plugin-install")


@dataclass
class Person:
    """What the person at Claude Code's terminal does with each thing it asks."""

    first_run: Literal["answers", "quits"] = "answers"  # quits: on its first screen, as /exit or Ctrl-C there
    key: Literal["uses", "declines"] = "uses"  # the API key question
    # The browser login a login screen or `auth login` opens. Not recorded: what `auth login` exits with when the
    # person abandons it, here 1, or comes back without one, here 0.
    login: Literal["completes", "abandons", "returns"] = "completes"
    plugin: Literal["accepts", "declines"] = "accepts"  # the [y/N] before a plugin's install command runs


@dataclass
class Config:
    """What one config directory holds besides its files: the login it was given, whether hands' plugin is installed,
    and where the marketplace it comes from was added from."""

    login: Login | None = None
    plugin: bool = False
    marketplace: Source | None = None
    # The provider `auth status` names. Every recording is of firstParty; another is that recording with this one field
    # changed, as a brain on Bedrock would answer.
    provider: str = "firstParty"


@dataclass
class Fake:
    person: Person = field(default_factory=Person)
    configs: dict[Path, Config] = field(default_factory=dict[Path, Config])
    files: dict[Path, bytes] = field(default_factory=dict[Path, bytes])
    # The mode each file was last replaced with.
    modes: dict[Path, int] = field(default_factory=dict[Path, int])
    # Files that are there and cannot be read, as a directory where a file should be.
    unreadable: set[Path] = field(default_factory=set[Path])
    # Whether a source added as a marketplace names a repository.
    repository: bool = True
    # Subcommands that cannot be run, as when `claude` is no program: each raises Unreachable.
    unrunnable: set[str] = field(default_factory=set[str])
    transcript: list[str] = field(default_factory=list[str])
    # Each run of `claude` hands asked for: which, the instance it ran as, and the files there as it started.
    runs: list["Ran"] = field(default_factory=list["Ran"])

    def config(self, claude: Instance) -> Config:
        return self.configs.setdefault(config_dir(claude.environment), Config())

    def told(self, said: str) -> None:
        """What hands tells the person, at the terminal Claude Code is about to ask at."""
        self.transcript.append(f"told: {said}")

    def _ran(self, name: str, claude: Instance) -> None:
        if name in self.unrunnable:
            raise Unreachable(f"`claude {name}` could not be run: no such program")
        self.transcript.append(name if claude.cwd is None else f"{name} in {claude.cwd}")
        self.runs.append(Ran(name, claude, frozenset(self.files)))

    def _credentials(self, claude: Instance) -> Mapping[str, str]:
        """The credentials in the environment Claude Code runs with: its own, with settings.json's env over it."""
        raw = self.read(config_dir(claude.environment) / "settings.json")
        written = {} if raw is None else cast(dict[str, dict[str, str]], json.loads(raw)).get("env", {})
        merged = {**claude.environment, **written}
        return {name: merged[name] for name in (TOKEN, firstrun.API_KEY) if merged.get(name)}

    def auth_status(self, claude: Instance) -> Answer:
        self._ran("auth status", claude)
        held = self.config(claude)
        credentials = self._credentials(claude)
        # A credential in the environment is said over a login stored in the config directory, as Claude Code uses it
        # first; each answer is the recording of that one alone.
        if TOKEN in credentials:
            said = answered("auth-status-oauth-token")
        elif firstrun.API_KEY in credentials:
            said = answered("auth-status-api-key-environment")
        else:
            match held.login:
                case "claude.ai":
                    said = answered("auth-status-claude-ai")
                case "console":
                    said = answered("auth-status-console")
                case None:
                    said = answered("auth-status-logged-out")
        status = json.loads(said.stdout)
        return Answer(said.exit, json.dumps({**status, "apiProvider": held.provider}, indent=2).encode() + b"\n", said.stderr)

    def plugins(self, claude: Instance) -> Answer:
        self._ran("plugin list", claude)
        return answered("plugin-list-hands" if self.config(claude).plugin else "plugin-list-none")

    def marketplaces(self, claude: Instance) -> Answer:
        self._ran("plugin marketplace list", claude)
        match self.config(claude).marketplace:
            case "github":
                return answered("marketplace-list-github")
            case "checkout":
                return answered("marketplace-list-checkout")
            case None:
                return answered("marketplace-list-none")

    def add_marketplace(self, claude: Instance, source: str) -> int:
        self._ran(f"plugin marketplace add {source}", claude)
        said = answered("marketplace-add-added" if self.repository else "marketplace-add-failed")
        self.transcript.append(f"shown: {(said.stdout + said.stderr).decode()}")
        if said.exit == 0:
            self.config(claude).marketplace = "github"
        return said.exit

    def install_plugin(self, claude: Instance, plugin: str) -> int:
        self._ran(f"plugin install {plugin}", claude)
        self.transcript.append(f"shown: {INSTALL['asked']}")
        match self.person.plugin:
            case "accepts":
                self.config(claude).plugin = True
                return 0
            case "declines":
                declined = cast(dict[str, object], INSTALL["declined"])
                return cast(int, declined["exit"])

    def auth_login(self, claude: Instance, method: Method | None) -> int:
        self._ran("auth login" if method is None else f"auth login --{method}", claude)
        match self.person.login:
            case "completes":
                self.config(claude).login = "console" if method == "console" else "claude.ai"
                return 0
            case "abandons":
                return 1
            case "returns":
                return 0

    def first_run(self, claude: Instance, settings: Mapping[str, object] | None) -> int:
        self._ran("first run" if settings is None else f"first run --settings {json.dumps(settings)}", claude)
        cwd = (claude.cwd or Path.cwd()).resolve()
        held = self.config(claude)
        state = firstrun.state_of(claude.environment, config_dir(claude.environment))
        raw = self.read(state)
        answers = firstrun.recorded_in(raw, state)
        key = firstrun.api_key(self, config_dir(claude.environment) / "settings.json", claude.environment)
        onboarded = isinstance(answers, firstrun.Recorded) and answers.finished
        trusted = isinstance(answers, firstrun.Recorded) and any(str(place) in answers.trusted for place in (cwd, *cwd.parents))
        key_open = key is not None and (not isinstance(answers, firstrun.Recorded) or key.value[-20:] not in answers.approved | answers.rejected)
        # The screens in the order the recording shows them: the API key question stands where the login screen would.
        asked = [
            *(("theme",) if not onboarded else ()),
            *(("login",) if not onboarded and key is None and held.login is None else ()),
            *(("api_key",) if key_open else ()),
            *(("notes",) if not onboarded else ()),
            *(("trust",) if not trusted else ()),
        ]
        # Where the person stops: on the first screen when they quit, on the login when they abandon it, else nowhere.
        stops = 1 if self.person.first_run == "quits" else asked.index("login") + 1 if "login" in asked and self.person.login != "completes" else None
        self.transcript.extend(f"shown: {SCREENS[screen]}" for screen in asked[:stops])
        if stops is not None:
            # A run quit part-way exits 0 as one through every screen, and leaves the state its start wrote.
            if raw is None:
                self.replace(state, json.dumps(STATES["started"]), 0o600)
            return 0
        if "login" in asked:
            pinned = None if settings is None else settings.get("forceLoginMethod")
            held.login = "console" if pinned == "console" else "claude.ai"
        written = {} if raw is None or isinstance(answers, firstrun.Blank) else cast(dict[str, object], json.loads(raw))
        responses = cast(dict[str, list[str]], written.get("customApiKeyResponses", {"approved": [], "rejected": []}))
        if key_open and key is not None:
            chosen = "approved" if self.person.key == "uses" else "rejected"
            responses = {**responses, chosen: [*responses.get(chosen, []), key.value[-20:]]}
        trusted_entry = cast(dict[str, object], cast(dict[str, object], STATES["trusted"]["projects"])["/work"])
        projects = cast(dict[str, object], written.get("projects", {}))
        onboarding = {name: STATES["onboarded"][name] for name in ("hasCompletedOnboarding", "lastOnboardingVersion")}
        made = {**STATES["started"], **written, **onboarding, "customApiKeyResponses": responses, "projects": {**projects, str(cwd): trusted_entry}}
        self.replace(state, json.dumps(made, indent=2), 0o600)
        return 0

    def read(self, file: Path) -> bytes | None:
        if file in self.unreadable:
            raise Rejected(f"{file} unreadable: [Errno 21] Is a directory")
        return self.files.get(file)

    def replace(self, file: Path, text: str, mode: int) -> None:
        self.files[file] = text.encode()
        self.modes[file] = mode

    def create(self, file: Path, text: str) -> bool:
        if file in self.files:
            return False
        self.files[file] = text.encode()
        return True


@dataclass(frozen=True)
class Ran:
    """One run of `claude`: which subcommand, the instance it ran as, and the files there as it started."""

    asked: str
    claude: Instance
    files: frozenset[Path]


def config_dir(environment: Mapping[str, str]) -> Path:
    """The config directory a Claude Code with this environment runs under."""
    named = environment.get("CLAUDE_CONFIG_DIR")
    return Path(named) if named else Path(environment.get("HOME") or Path.home()) / ".claude"


def onboard(fake: Fake, brain: Path, settings: bytes = b'{"syncClaudeAiSkills": false, "syncClaudeAiPlugins": false}', trusted: bool = True, login: Login | None = "claude.ai") -> None:
    """A brain home Claude Code has been through its onboarding on, holding `settings` and `login`; the directory it runs
    in trusted, as Claude Code trusts it, by a directory above it, unless not `trusted`."""
    projects = {str(brain.resolve().parent): {"hasTrustDialogAccepted": trusted}}
    fake.files[brain / ".claude.json"] = json.dumps({"hasCompletedOnboarding": True, "projects": projects}).encode()
    fake.files[brain / "settings.json"] = settings
    fake.configs[brain] = Config(login=login)
