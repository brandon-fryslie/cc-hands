"""The brain's process: one long-lived, slim Claude Code on its own login, behind hands' proxy, reaching hands over MCP.

It is Claude Code as anyone runs it: interactive, on a terminal hands holds, under fritter, and asked nothing a person
at its keyboard could not ask. A turn is typed into its input and sent with Return, and an interrupt is Escape. Nothing
else is typed into it: its input is the user's, and what hands asks in the background is asked of a Claude Code of its
own (`hands.brain.asides`). What it says is read from the wire, not from its screen [the design's rule: primary facts
from the wire, derivative ones from the harness], and the harness is heard only through the hooks it posts to a
listener of hands' own: that a typed turn was taken, and that it ended, or that the API failed it.

Its login, settings, and skills live in a directory hands owns, set up once by `hands login`, which runs Claude Code's
own first run there, as any Claude Code is set up, on any login Claude Code takes, and it runs in that directory's empty cwd, never in a project. What it may use, what it may do without asking,
and which MCP servers it has are that directory's to say, as they are for any Claude Code: its settings.json and its
.claude.json. hands adds only its own server, its hooks, and the skills it ships for the brain's own jobs, and keeps out
what the login brings from the account.
"""

import asyncio
import contextlib
import glob
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from aiohttp import web
from loguru import logger

from hands.brain.mcp import SERVER_NAME
from hands.core.effects import Allow, Deny, Text
from hands.core.session import Permission, PromptText, SessionId, pasted
from hands.core.wire import MainTurn, Observed, Sent, tool_names
from hands.core.trace import Span
from hands.sessions.audit import Record
from hands.sessions.hookconfig import PERMISSION_DEADLINE_SECONDS, declared
from hands.sessions.hooks import called, hook_output
from hands.sessions import firstrun
from hands.sessions.files import replace_whole
from hands.sessions.payload import Payload, Rejected
from hands.sessions.pseudoterminal import ClaudeCode, on_terminal
from hands.sessions.typing import Typist, Untyped
from hands.sessions.untap import untapped
from hands.sessions.wide import Begun, annotate, begun, continuing, fail, here, unit
from hands.sessions.wrapper import Unpackaged, packaged, real_claude
from hands.voice.backends import Account

# What --bare would have switched off, switched off one by one so the OAuth login stays on (hands-wire-6ic.8wu, 2.1.284).
# LSP needs no switch: it comes only from plugins, the brain's own setup installs none, and hands' own declares none.
SLIM = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION": "false",
    # A background command's notification opens a turn of its own, under a prompt id no turn hands typed carries, which
    # would take a typed turn's place or leave it never ending (hands-wire-6ic.99l review).
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    # A scheduled prompt opens a turn of its own the same way when it fires (hands-brain-d8g.9d6 review, 2.1.288).
    "CLAUDE_CODE_DISABLE_CRON": "1",
    # The account's claude.ai connectors are Brandon's, never the brain's: loaded, they joined every request after the
    # first (turn 2 grew from 8.7 KB to 112 KB; hands-wire-6ic.eph, 2.1.284). The skills and plugins the account syncs
    # are turned off in the brain's settings.json, the only place Claude Code reads that switch from (2.1.288).
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}

# The tool Claude Code offers in place of the MCP tools it defers.
TOOL_SEARCH = "ToolSearch"

# The skills hands gives the brain, as a plugin of its own: shipped with the code that runs the jobs they are for, beside
# the skills of the brain's own setup and never in it.
PLUGIN = Path(__file__).parent / "plugin"
# Each named as the brain calls it, plugin:skill, so it may use them without asking, as it may hands' tools.
PLUGIN_SKILLS = tuple(
    f"Skill({json.loads((PLUGIN / '.claude-plugin' / 'plugin.json').read_text())['name']}:{skill.parent.name})" for skill in sorted((PLUGIN / "skills").glob("*/SKILL.md"))
)

# What Claude Code prefers to its own login: credentials, and the switches that send it to a provider other than
# Anthropic's API (2.1.289). The brain's login is its config directory's, whichever Claude Code takes, a key in its
# settings.json among them; inherited from hands' environment, any of these would put the brain on another account or
# provider without a word, so none is passed on.
FOREIGN_LOGINS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    *(f"CLAUDE_CODE_USE_{provider}" for provider in ("BEDROCK", "VERTEX", "FOUNDRY", "ANTHROPIC_AWS", "ANTHROPIC_GOOGLE_CLOUD", "MANTLE", "GATEWAY")),
)

# A dialog nobody can see would hold its turn open forever with no hook to say so (2.1.288). A permission its setup asks
# about is held at its hook while the user is asked by voice, in a turn of theirs; anything else is answered no there,
# and the brain hears why.
NOBODY = "Nobody can be asked: no turn of the user's is in flight to ask in, so what this Claude Code's settings would ask about is refused."
UNREAD = "hands could not read this permission request, so it is refused."
BROKEN = "hands failed while it settled this permission request, so it is refused."
UNVOICED = "Nobody sees this dialog: ask the user in your reply instead, and they will answer in their next turn."
UNANSWERED = "No answer came from the user in time, so it did not run. Tell them aloud that it did not run and what that leaves undone."
SPOKEN_OVER = "The user spoke over this turn before they could be asked, so it did not run."
STOPPED = "The user stopped this turn, so it did not run."
DECLINED: Mapping[str, object] = {"hookSpecificOutput": {"hookEventName": "Elicitation", "action": "decline"}}

# What the brain posts to hands, each to its own path: a typed turn taken, a turn ended, a turn the API failed, and a
# dialog about to open. Escape ends a turn with none of them (measured on 2.1.285), so a turn told to stop is over when
# it is told.
HOOKS = ("UserPromptSubmit", "Stop", "StopFailure", "PermissionRequest", "Elicitation")
# A paste of four lines or more, as a UserPromptSubmit hook's `prompt` holds it: what was pasted, with a line end added
# where it had none, inside a tag pair (2.1.289, measured 2026-10-04).
PASTE = re.compile(r'\n\n<pasted_content id="(\w+)">\n(.*)</pasted_content id="\1">\n', re.DOTALL)

# `claude auth status` answers in about a second; one that has not answered in this long is not going to.
AUTH_STATUS_SECONDS = 20.0
# How long fritter has to start the brain and open its socket, and the brain to turn its input on.
START_SECONDS = 10.0
# How long a typed turn has to be taken: a cold start loads the MCP server before the input is read.
TAKE_SECONDS = 30.0
# How many times a turn is typed at most: once, and once more when a prompt that is no turn's ran through its wait and
# was stopped, which leaves nothing ahead of it.
TYPINGS = 2
# How far apart two Ctrl-Cs are pressed into the brain: Claude Code exits on a second within 800ms of one that found its
# input empty.
EXIT_SECONDS = 1.0


def _kept(prompt: str) -> str:
    """A prompt as Claude Code keeps what was typed: out of a paste's tags, and without its trailing whitespace, which
    Claude Code trims from what it takes (2.1.289, measured 2026-10-04)."""
    paste = PASTE.fullmatch(prompt)
    return (prompt if paste is None else paste[2]).rstrip()


@dataclass(frozen=True)
class Station:
    """What every slim Claude Code of hands' own runs on, the brain and each one a side question is asked of."""

    config_dir: Path
    cwd: Path
    model: str
    proxy_url: str
    # hands' own environment, which each is started in less what environment() keeps out; kept out of the repr, as it
    # holds keys, and out of the station's identity, which it is not part of.
    inherited: Mapping[str, str] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Launch:
    """Everything the brain is started with."""

    station: Station
    # The account its config directory is logged in as, read as the run started.
    account: Account
    instruction: str
    mcp_config: str
    conversation: "Conversation"

    @property
    def session(self) -> SessionId:
        return self.conversation.session


@dataclass(frozen=True)
class Fresh:
    """A conversation begun new, under a session chosen by hands; `untranscribed` is the one held before it, when there
    was one, which Claude Code has no transcript of: it never had a turn (Claude Code writes none until one), or its
    transcript was cleaned away."""

    # [LAW:one-source-of-truth] chosen by hands, so the brain's requests are known as its own from the first one on the
    # wire, and its hooks from the first one posted.
    session: SessionId
    untranscribed: SessionId | None


@dataclass(frozen=True)
class Resumed:
    """The conversation the brain held when hands last ran, taken up again under its own session: a restart, an upgrade,
    or a change of model goes on from what was said, compacted as Claude Code compacts any conversation."""

    session: SessionId


Conversation = Fresh | Resumed


def held(config_dir: Path) -> Path:
    """The session of the brain's conversation, written by hands alone as each brain on `config_dir` starts."""
    return config_dir / "conversation"


def conversation(config_dir: Path, new: SessionId) -> Conversation:
    """The conversation a brain on `config_dir` starts in: the one held, while Claude Code still has its transcript, or
    else one begun under `new`."""
    session = _holding(config_dir)
    if session is None:
        return Fresh(new, None)
    # [LAW:one-source-of-truth] Claude Code's own transcript says whether there is a conversation to resume: named by
    # its session, under any project directory, as Claude Code finds the session it resumes (2.1.289, measured).
    if glob.glob(glob.escape(str(config_dir / "projects")) + f"/*/{glob.escape(session)}.jsonl"):
        return Resumed(session)
    return Fresh(new, session)


def hold(config_dir: Path, session: SessionId) -> None:
    """Remember `session` as the brain's conversation, replacing whatever was held whole."""
    path = held(config_dir)
    written = path.with_name(f"{path.name}.writing")
    written.write_text(f"{session}\n")
    written.replace(path)


def let_go(config_dir: Path) -> SessionId | None:
    """Forget the brain's conversation, so the next brain on `config_dir` begins a new one: the session that was held,
    when one was."""
    session = _holding(config_dir)
    held(config_dir).unlink(missing_ok=True)
    return session


def _holding(config_dir: Path) -> SessionId | None:
    try:
        return SessionId(held(config_dir).read_text().strip())
    except FileNotFoundError:
        return None


def slim(claude: Path, model: str, conversation: Conversation) -> list[str]:
    """A slim Claude Code's command line: the real claude, interactive, reading no settings but its config directory's."""
    return [
        str(claude),
        "--model", model,
        *_conversing(conversation),
        # The config directory's settings.json is the only settings file read: none from the working directory.
        "--setting-sources", "user",
    ]


def _conversing(conversation: Conversation) -> tuple[str, str]:
    match conversation:
        case Fresh(session=session):
            return ("--session-id", session)
        case Resumed(session=session):
            # Claude Code keeps the session it resumes unless told to fork it (2.1.289, --fork-session).
            return ("--resume", session)


def command(launch: Launch, claude: Path, hooks: str) -> list[str]:
    """The brain's command line: a slim Claude Code on its own setup, plus hands' server, instruction, skills, and hooks posted to the listener at `hooks`."""
    return [
        *slim(claude, launch.station.model, launch.conversation),
        # The instruction as this hands writes it on every request, a resumed conversation's too: Claude Code otherwise
        # sends the system prompt its conversation first had until it is compacted (2.1.289, measured: a resume launched
        # with a new --append-system-prompt sent the old one). Rendered fresh, it is byte-identical request to request.
        "--system-prompt-snapshot", "off",
        # Beside the MCP servers its own setup names, never in place of them.
        "--mcp-config", launch.mcp_config,
        # After Claude Code's own system prompt, never in place of it: the API checks that it opens as Claude Code's does.
        "--append-system-prompt", launch.instruction,
        # hands' tools are how the brain reaches the sessions at all, and its skills how it does its jobs there; a deny rule
        # in its own setup still outranks this.
        "--allowedTools", f"mcp__{SERVER_NAME}", *PLUGIN_SKILLS,
        "--plugin-dir", str(PLUGIN),
        "--settings", json.dumps({
            # [LAW:single-enforcer] each hook declared as the plugin declares it: a held permission's lives as long as a
            # working session's, and is denied by the same deadline.
            "hooks": {event: [{"hooks": [{"type": "http", "url": f"{hooks}/{event}", **declared(event)}]}] for event in HOOKS},
            # Tool search defers MCP tools behind ToolSearch, so hands' tools would reach the model only after a round trip
            # of their own on a spoken turn. Set here, it outranks the brain's settings.json env and hands' environment,
            # which a process environment does not (2.1.288, measured behind a localhost base URL, hands-brain-8gb).
            "env": {"ENABLE_TOOL_SEARCH": "false"},
        }),
    ]


def environment(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> dict[str, str]:
    """A slim Claude Code's environment: hands' own, less anything that would stand in for the login in `config_dir`, reaching the API at `base_url`."""
    # [LAW:one-source-of-truth] the brain is the same brain wherever hands was started: a daemon started inside a tapped
    # session does not hand the brain that session's tap.
    kept = {name: value for name, value in untapped(inherited).items() if name not in FOREIGN_LOGINS}
    return {**kept, **SLIM, "CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_BASE_URL": base_url}


def workdir(config_dir: Path) -> Path:
    """The empty directory a slim Claude Code on the login in `config_dir` runs in, made if it is not there: never a project."""
    cwd = _cwd(config_dir)
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


def _cwd(config_dir: Path) -> Path:
    return config_dir / "cwd"


async def spawn(station: Station, argv: Sequence[str]) -> ClaudeCode:
    """Run a slim Claude Code's command on a terminal of hands' own, in its own directory and environment."""
    return await on_terminal(argv, station.cwd, environment(station.config_dir, station.proxy_url, station.inherited))


class BrainGone(Exception):
    """The brain ended, or cannot be typed into, while a turn waited on it, or before one was asked."""


class Untaken(Exception):
    """The brain was typed a turn and never took it: it is on a screen that is not its input."""


class NotLoggedIn(Exception):
    """The brain's config directory holds no login whose requests reach hands' proxy, so every turn would fail or go unheard."""


class LoginFailed(Exception):
    """Claude Code's own login did not finish; whatever login the brain held before, it still holds."""


class Unstartable(Exception):
    """A Claude Code of hands' own could not be started: no claude to run, or a fritter that never opened its socket, or a brain that never turned its input on."""


def brain_claude(inherited: Mapping[str, str]) -> Path:
    """The Claude Code hands runs as its own: the real claude on PATH, past every hands shim, which would run it as a session."""
    claude = real_claude(inherited.get("PATH", ""))
    if claude is None:
        raise Unstartable("no claude on PATH but hands' shims, so there is no Claude Code for hands to run as its own")
    return claude


# The logins `claude auth login` makes, each named by its own flag: a Claude plan, or an Anthropic Console key.
type Method = Literal["claudeai", "console"]

# [LAW:one-source-of-truth] the authMethod `claude auth status` says of each login once it is made (2.1.289).
AUTH_METHODS: dict[Method, str] = {"claudeai": "claude.ai", "console": "api_key"}
# What Claude Code's login screen calls each.
AUTH_PICKED: dict[Method, str] = {"claudeai": "a Claude plan", "console": "an Anthropic Console account"}


@dataclass(frozen=True)
class Kept:
    """The login the brain held already, kept by a `hands login` that asked for none in particular: no Claude Code ran."""

    account: Account


@dataclass(frozen=True)
class FirstRun:
    """The brain's first run of Claude Code: why it took one, and the login it was pinned to, None when the brain held
    one already, which a pin to another would have Claude Code refuse."""

    why: str
    pinned: Method | None


@dataclass(frozen=True)
class Made:
    """A login Claude Code made at this terminal: through the brain's first run, when there was one, and through
    `claude auth login` with `auth_login`, when that ran after or instead."""

    account: Account
    first_run: FirstRun | None
    auth_login: Method | None


# What `hands login` left the brain holding.
type Login = Kept | Made


def unanswered(config_dir: Path) -> firstrun.Unanswered | None:
    """What the brain would start on of Claude Code's first screens, or None when it would not, as Claude Code records
    them in the brain's own .claude.json. A config directory made by anything else, `claude auth status` among them, has
    none of them. Its API key is the one its settings.json sets: hands' never reaches it (FOREIGN_LOGINS). Raises
    Unstartable when either file is there but cannot be read, which no first run mends."""
    try:
        return firstrun.unanswered(firstrun.recorded(config_dir / ".claude.json"), _cwd(config_dir).resolve(), firstrun.api_key(config_dir / "settings.json", {}))
    except Rejected as error:
        raise Unstartable(f"the brain's first-run state could not be read: {error}") from None


def onboarded(config_dir: Path) -> bool:
    """Answers, in the brain's own .claude.json, the first screens that are only the brain's preferences: its onboarding
    (a theme, for a terminal nobody reads, and Claude Code's notes) finished, and the directory it runs in trusted. Whether
    it wrote. Left to Claude Code's first run: a login, an API key its settings.json sets, and a .claude.json that records
    nothing hands can read, which only that run writes over (firstrun.recorded). Raises Rejected when the file is there but
    cannot be read at all."""
    state = config_dir / ".claude.json"
    # [LAW:one-source-of-truth] the directory firstrun.unanswered reads trust for, as Claude Code records it, resolved.
    cwd = _cwd(config_dir).resolve()
    # One read: the bytes the answers are judged on are the ones they are merged into.
    raw = firstrun.read(state)
    answers = firstrun.recorded_in(raw, state)
    # [LAW:single-enforcer] whether either is open is firstrun's to say, the trust of a directory above among it.
    if firstrun.unanswered(answers, cwd, None) is None or (raw is not None and isinstance(answers, firstrun.Blank)):
        return False
    written: Mapping[str, object] = {} if raw is None else Payload.parse(raw).fields
    projects = Payload.of(written.get("projects", {}), "its projects").fields
    # An entry that is no object is no trust, as firstrun reads it, and Claude Code's first run would write it again.
    entry = projects.get(str(cwd))
    place = cast(dict[str, object], entry) if isinstance(entry, dict) else {}
    made = {**written, "hasCompletedOnboarding": True, "projects": {**projects, str(cwd): {**place, "hasTrustDialogAccepted": True}}}
    # 0600, as Claude Code keeps it: it holds the login's account.
    replace_whole(state, json.dumps(made, indent=2), 0o600)
    return True


def answered(config_dir: Path) -> None:
    """Raises Unstartable, naming the command that answers them, while the brain would start on Claude Code's first screens."""
    if (open_ := unanswered(config_dir)) is not None:
        raise Unstartable(f"the brain has not been through Claude Code's first screens ({open_.why}): `hands login` answers them")


class Unasked(Exception):
    """Claude Code had something to ask the person, and `hands login`'s input is no terminal it could ask at."""


def login(config_dir: Path, base_url: str, inherited: Mapping[str, str], method: Method | None, terminal: bool, starting: Callable[[str], None]) -> Login:
    """Log `config_dir` in with Claude Code's own login, at this terminal, in the directory the brain runs in.

    Two steps, as the person's own first run takes them. The first screens that are only the brain's preferences, its
    theme and the trust of its directory, hands answers itself (onboarded). One still unanswered after, an API key its
    settings.json sets or a state hands cannot read, gets Claude Code's first run, answered once here: `claude auth login`
    alone leaves it for the brain's first start, where nobody is at its keyboard. On a brain holding no login, that run is
    pinned to the login made, so it makes it. Then
    `claude auth login` runs while the brain holds no login, or one `method` asks for that the first run did not just
    make: with none asked for, a login the brain holds is kept, and one it lacks is Claude Code's own default, a Claude
    plan. `terminal` is whether the person is at one to be asked; `starting`
    is told what Claude Code is about to ask, just before it asks. Raises LoginFailed when Claude Code did not finish, or
    left the brain on another login than the one asked for."""
    # Before any run of Claude Code on it, which would sync what the brain's settings do not keep out.
    account_kept_out(config_dir)
    # [LAW:nothing-unseen] whether hands answered the brain's preference screens, before what is left of its first run is
    # judged: asking nothing, it needs no terminal.
    try:
        annotate(onboarded=onboarded(config_dir))
    except Rejected as error:
        raise Unstartable(f"the brain's first-run state could not be read: {error}") from None
    # [LAW:one-source-of-truth] the brain's own claude, environment, and settings sources, so the login lands in its config
    # directory, which the daemon reads, no credential of this shell's stands in for the one being made, and the first
    # run's screens are answered under the settings the brain starts with. Resolved before anything is said of them.
    claude, env, cwd = brain_claude(inherited), environment(config_dir, base_url, inherited), workdir(config_dir)
    first_run = unanswered(config_dir)
    held = holding(config_dir, base_url, inherited)
    # Asked for none in particular, whatever login a brain through its first run holds is the brain's.
    if method is None and held is not None and first_run is None:
        return Kept(held)
    if not terminal:
        # Claude Code with no terminal to read answers a prompt instead of asking, and its login waits on a code.
        raise Unasked("Claude Code asks for the brain's login at a terminal, and this command's input is not one")
    # [LAW:one-source-of-truth] the login made: the one asked for, or a Claude plan, Claude Code's own default (2.1.289).
    made: Method = method or "claudeai"
    other = "" if method else ": `hands login --console` logs it in with an Anthropic Console account instead"
    # forceLoginMethod takes the first run past Claude Code's choice of login, straight to the one made; Claude Code
    # refuses a login held that it does not name, so a brain holding one is never pinned, and `claude auth login` below
    # moves it to another (2.1.289).
    ran: FirstRun | None = None
    if first_run is not None:
        ran = FirstRun(first_run.why, None if held is not None else made)
        login_asked = "" if ran.pinned is None else f" A login it asks for is for {AUTH_PICKED[ran.pinned]}{other}."
        starting(f"Claude Code now starts as the brain, hands' own Claude Code, in {cwd}, and asks what it asks only once: {first_run.listed}.{login_asked} Answer each, then type /exit")
        pin = {} if ran.pinned is None else {"forceLoginMethod": ran.pinned}
        _ran(subprocess.run([claude, "--setting-sources", "user", "--settings", json.dumps(pin)], cwd=cwd, env=env), "the brain's first run of Claude Code")
        # A first run quit before its last screen exits 0 as one that answered them all.
        if (left := unanswered(config_dir)) is not None:
            raise LoginFailed(f"the brain's first run of Claude Code was quit before its last screen ({left.why})")
        held = holding(config_dir, base_url, inherited)
    # A login asked for is made again unless the first run just made it.
    if held is not None and (method is None or (ran is not None and ran.pinned is not None and held.method == AUTH_METHODS[method])):
        return Made(held, ran, None)
    starting(f"Claude Code now logs the brain in with its own login: it opens your browser, or prints a link to open, for {AUTH_PICKED[made]}{other}")
    _ran(subprocess.run([claude, "auth", "login", f"--{made}"], cwd=cwd, env=env), "`claude auth login` for the brain")
    if (account := holding(config_dir, base_url, inherited)) is None:
        raise LoginFailed("`claude auth login` for the brain finished, and the brain holds no login")
    # [LAW:no-silent-failure] a login its settings.json sets ahead of the one made can leave the brain on another.
    if method is not None and account.method != AUTH_METHODS[method]:
        raise LoginFailed(f"the brain holds {account}, not the {AUTH_METHODS[method]} login asked for")
    return Made(account, ran, made)


def _ran(run: subprocess.CompletedProcess[bytes], what: str) -> None:
    if run.returncode != 0:
        raise LoginFailed(f"{what} exited {run.returncode}")


# Each is on unless settings.json says false: they sync the login's account's skills and plugins, which are Brandon's.
KEPT_OUT = ("syncClaudeAiSkills", "syncClaudeAiPlugins")


def starting_settings(config_dir: Path) -> bool:
    """Writes the settings.json a brain home starts with when it has none; whether it wrote one. One that is there is
    left as it is: the brain's settings are its directory's to say."""
    # [LAW:one-source-of-truth] what account_kept_out requires, and the mode docs/guide.md says the brain runs in.
    starting = {**dict.fromkeys(KEPT_OUT, False), "permissions": {"defaultMode": "default"}}
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (config_dir / "settings.json").open("x") as made:
            json.dump(starting, made, indent=2)
    except FileExistsError:
        return False
    return True


def account_kept_out(config_dir: Path) -> None:
    """Raises Unstartable unless the brain's settings.json keeps out the skills and plugins its login's account syncs.

    They are Brandon's, never the brain's, and settings.json is the only place Claude Code reads either switch from
    (2.1.288): not --settings, and not the environment. Left out, each is on."""
    settings = config_dir / "settings.json"
    fix = f'set "syncClaudeAiSkills": false and "syncClaudeAiPlugins": false in {settings}'
    try:
        said = Payload.parse(settings.read_bytes())
        synced = [switch for switch in KEPT_OUT if said.fields.get(switch) is not False]
    except (OSError, Rejected) as error:
        raise Unstartable(f"the brain's settings could not be read ({error}): {fix}") from None
    if synced:
        raise Unstartable(f"the brain would load its account's {' and '.join(synced)}: {fix}")


def logged_in(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> Account:
    """The account `config_dir` is logged in as; raises NotLoggedIn, naming the command that makes a login, when it has
    none, or one whose requests do not reach hands' proxy."""
    if (account := holding(config_dir, base_url, inherited)) is None:
        raise NotLoggedIn("the brain has no login: `hands login` gives it one")
    return account


def holding(config_dir: Path, base_url: str, inherited: Mapping[str, str]) -> Account | None:
    """The account `config_dir` is logged in as, or None when it holds no login; raises NotLoggedIn when `claude auth
    status` cannot say, or says it holds one whose requests do not reach hands' proxy, none of which a login mends."""
    try:
        # A timed-out child is killed and reaped by run itself.
        asked = subprocess.run(
            [brain_claude(inherited), "auth", "status"],
            env=environment(config_dir, base_url, inherited),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=AUTH_STATUS_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise NotLoggedIn(f"`claude auth status` for the brain did not answer in {AUTH_STATUS_SECONDS:.0f}s") from None
    try:
        status = Payload.parse(asked.stdout)
        if not status.flag("loggedIn"):
            return None
        # [LAW:no-silent-failure] hands' proxy, which hears what the brain says, forwards to Anthropic's API alone, so a
        # brain on another provider would answer nothing. hands' environment never chooses one (FOREIGN_LOGINS).
        if (provider := status.text("apiProvider")) != "firstParty":
            raise NotLoggedIn(f"the brain reaches Claude through {provider}, not the Anthropic API hands' proxy forwards to: {config_dir / 'settings.json'} or the machine's managed settings choose {provider}")
        return Account(status.text("authMethod"), status.optional_text("email") or status.optional_text("apiKeySource"))
    except Rejected as error:
        raise NotLoggedIn(f"`claude auth status` for the brain answered {asked.stdout[:200]!r} {asked.stderr[:200]!r}, not its status: {error}") from None


@dataclass(frozen=True)
class BrainAnswered:
    """The end of a brain turn: its Stop hook, its StopFailure hook with what failed it, or the Escape hands pressed to
    stop it. Its words are on the wire."""

    prompt: str  # the prompt id Claude Code gave the turn, which every hook of the turn and its transcript records carry
    error: str | None


# Compared by identity: each is its own request, however alike two calls are.
@dataclass(frozen=True, eq=False)
class Asked:
    """A permission the brain's own setup asks about, held at its hook while the user is asked: settled once, by what they
    answer, by nobody answering in time, or by the turn's end, whichever comes first."""

    permission: Permission
    decision: "asyncio.Future[Allow | Deny]"

    @property
    def open(self) -> bool:
        return not self.decision.done()

    def settle(self, decision: Allow | Deny) -> None:
        # The first to settle it answers its hook; any later one comes after that answer went.
        if self.open:
            self.decision.set_result(decision)


@dataclass(frozen=True)
class _Posted:
    """A hook the brain posted, to the path of its event, as it was posted, and the body its post is answered with."""

    event: str
    body: bytes
    reply: "asyncio.Future[Mapping[str, object]]"


@dataclass
class _Turn:
    answered: asyncio.Future[BrainAnswered]
    taken: asyncio.Future[str]  # the prompt id Claude Code gave the turn when it first took it
    # Resolved once the turn's typing is over, however it went: then it is taken, or never will be.
    tried: asyncio.Future[None]
    # Told of each permission the turn holds at its hook, to put it to the user.
    asks: Callable[[Asked], None]
    # The turn's own unit of work, begun as it was asked: each dialog it holds is a part of it.
    began: Begun
    # How long the turn has to be taken, read once as it is asked: each typing of it and its stop wait the same time.
    take: float
    # What the turn types into the brain's input.
    typed: PromptText
    # The tools the turn's latest request offered the model: what the brain's own setup gave it, beside hands' tools.
    offered: tuple[str, ...] = ()
    # The prompts Claude Code took while the turn waited to be taken that were not the turn's own, and those that were no
    # turn's its typing or its stop stopped.
    others: tuple[str, ...] = ()
    # How many times the turn was typed, up to TYPINGS.
    typings: int = 0
    # Each prompt Claude Code took the turn as: two when it was typed again after an Escape that may have stopped the
    # first, which then posts no Stop. The turn is over at the Stop of either.
    prompts: tuple[str, ...] = ()
    # What broke in the brain's own work for the turn: it ends with this once Claude Code's turn is stopped.
    broken: BaseException | None = None


class Brain:
    """A running brain: a turn is asked with `ask`, which returns once the brain has said the turn is over."""

    def __init__(
        self,
        claude: ClaudeCode,
        session: SessionId,
        typist: Typist,
        hooks: "asyncio.Queue[_Posted]",
        listener: web.AppRunner,
        sockets: Path,
        config_dir: Path,
        record: Record,
        launched: Span,
    ) -> None:
        self._claude = claude
        self._config_dir = config_dir
        self.session = session
        self._typist = typist
        self._listener = listener
        self._sockets = sockets
        self._record = record
        # What the brain does that no turn asked for is a part of its launch, in the launch's trace.
        self._launched = launched
        # [LAW:single-enforcer] the input is the user's, and only two things are ever typed into it: a turn, and the
        # keys that stop one. A turn keeps it until its hook says Claude Code took the turn, and a stop's keys go in alone.
        self._input = asyncio.Lock()
        # [LAW:no-ambient-temporal-coupling] the turn in flight is the brain's own state, not its asker's: it is over
        # when its hook says so, whether or not anyone still waits on it, and the next is typed only then.
        self._turn: _Turn | None = None
        # The permissions held at their hooks now: one for each call of a reply that asks. Nothing is typed into the brain
        # while one is held but the Escape that stops its turn: what is typed goes to the dialog Claude Code draws, whose
        # Return says yes (2.1.288, spike hands-brain-d8g.aur). A turn is typed only once the last has ended, and the stage
        # presses no Escape while the user is being asked.
        self._held: set[Asked] = set()
        self._typing: set[asyncio.Task[None]] = set()
        # The prompts Claude Code took that are no turn's, from their UserPromptSubmit until their Stop or the Escape that
        # stops them: one typed for a turn that had ended untaken, or a task's notification.
        # In the order Claude Code took them.
        self._unowned: dict[str, None] = {}
        # When the last stop pressed Ctrl-C, on the event loop's clock.
        self._cleared = float("-inf")
        with continuing(launched):
            self._heard = asyncio.ensure_future(self._hear_hooks(hooks))
            self._exit = asyncio.ensure_future(self._run_out(begun()))

    @property
    def pid(self) -> int:
        return self._claude.pid

    async def ask(self, text: str, asks: Callable[[Asked], None]) -> BrainAnswered:
        """One turn of the brain's, from its typing to its end; `asks` is told of each permission the turn holds for the user."""
        while self._turn is not None:
            await asyncio.wait({self._turn.answered})
        if self._exit.done():
            raise BrainGone(f"the brain had exited ({self._exit.result()}) before it was asked")
        loop = asyncio.get_running_loop()
        # Behind a space, as every prompt hands types: a leading / or ! is then the character it is.
        turn = self._turn = _Turn(loop.create_future(), loop.create_future(), loop.create_future(), asks, begun(), TAKE_SECONDS, Text(pasted(text)).typed)
        # An asker that stops waiting leaves the turn to be typed and to run to its end, which is still the brain's to hear.
        self._keep(self._send(turn), turn)
        return await asyncio.shield(turn.answered)

    def _keep(self, work: Coroutine[None, None, None], turn: _Turn | None) -> None:
        """Runs on in a task of the brain's own, whoever asked for it, for `turn`, the turn in flight it was begun in."""
        task = asyncio.create_task(work)
        self._typing.add(task)
        task.add_done_callback(lambda done: self._kept(done, turn))

    def _kept(self, task: "asyncio.Task[None]", turn: _Turn | None) -> None:
        self._typing.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._broke(turn, error)

    @contextlib.contextmanager
    def _done_for(self, turn: _Turn | None) -> Generator[None]:
        """The brain's own work done for `turn`, or for no turn: what it raises that nothing in it expected is `_broke`'s."""
        try:
            yield
        except Exception as error:
            self._broke(turn, error)

    def _broke(self, turn: _Turn | None, error: BaseException) -> None:
        """[LAW:single-enforcer] what the brain's own work raised that nothing in it expected: said once, with where it
        came from, and the end of the turn it was done for, stopped as an Escape stops it, so no dialog of it is left open
        for the next turn's Return, its asker is answered, and the turns behind it are typed."""
        logger.opt(exception=error).error(f"the brain's own work failed: {error!r}")
        if turn is not None and turn.broken is None:
            turn.broken = error
            self._keep(self._stop(turn), turn)

    async def _send(self, turn: _Turn) -> None:
        # [LAW:nothing-unseen] the turn is one event of the brain's own, from its typing to its end, however long its asker
        # still waits on it: a turn left to run on is still heard to its end.
        with unit("brain.turn", self._record, began=turn.began):
            try:
                # [LAW:no-ambient-temporal-coupling] the input is the turn's from its first key until Claude Code says it
                # took the turn. Claude Code reads keys that reach it together as one paste, and a Return inside a paste
                # sends nothing: what was typed 10ms behind a turn joined the turn's prompt, or was left in the input with
                # it (2.1.286, measured 2026-09-30, 2 of 4; none of 8 typed once the turn was taken).
                try:
                    async with self._input:
                        taken = await self._taking(turn)
                finally:
                    turn.tried.set_result(None)
                if not taken:
                    raise Untaken(f"the brain did not take the turn typed into it {turn.typings} time(s), {turn.take:g}s each; if it is on one of Claude Code's first screens, `hands login` answers them")
            except (BrainGone, Untaken) as error:
                self._over(turn, error)
            await asyncio.wait({turn.answered})
            annotate(prompts=turn.prompts, typings=turn.typings, offered=turn.offered, others=turn.others)
            match turn.answered.exception():
                case None:
                    if (error := turn.answered.result().error) is not None:
                        fail(f"the brain's turn ended in error: {error}")
                case failed:
                    fail(f"the brain failed the turn: {failed}")

    async def _taking(self, turn: _Turn) -> bool:
        """Types `turn`, with the input held, and waits `turn.take` seconds for it to be taken, or to end: says whether it
        was. [LAW:single-enforcer] a prompt Claude Code took that is no turn's runs ahead of what is typed behind it, for
        an ask that already failed or one nobody made: it is stopped before the turn is typed, and when one ran through the
        turn's wait, it is stopped and the turn typed once more, with its whole time and nothing left ahead of it. The
        turn may have been taken as that Escape went, which then stopped it instead: its typing again is still taken."""
        while True:
            if self._unowned:
                await self._clear(turn)
                if turn.answered.done():
                    return True
            turn.typings += 1
            await self._type(lambda typist: typist.type(turn.typed))
            await asyncio.wait({turn.taken, turn.answered}, timeout=turn.take, return_when=asyncio.FIRST_COMPLETED)
            if turn.taken.done() or turn.answered.done():
                return True
            if turn.typings == TYPINGS or not self._unowned:
                return False

    async def _clear(self, turn: _Turn) -> None:
        """Stops whatever Claude Code runs and empties its input, with the input held, for `turn`: the prompts that were
        no turn's it stopped are its `others`. Escape posts no hook for the prompt it stops, so they are no longer running."""
        stopped = tuple(self._unowned)
        await self._type(lambda typist: typist.press("escape"))
        # Escape refuses a permission held at its dialog and leaves its hook unanswered for good (2.1.288, spike
        # hands-brain-d8g.aur): it is settled here, and nothing can answer it later.
        self._settle(Deny(STOPPED))
        # Escape puts a prompt stopped before any reply back in the input, and with it what was typed queued behind the
        # prompt (2.1.289: its cancel pops the queue into the input), which the next turn typed would join; Ctrl-C clears
        # it. On an input left empty it arms Claude Code's exit instead, which a second Ctrl-C within 800ms takes,
        # whatever was typed between (2.1.285, read from its source 2026-09-30); so no two are pressed that close,
        # counted from when each went and with room for Claude Code to read the first late.
        loop = asyncio.get_running_loop()
        await asyncio.sleep(self._cleared + EXIT_SECONDS - loop.time())
        await self._type(lambda typist: typist.press("ctrl_c"))
        self._cleared = loop.time()
        for prompt in stopped:
            # One may have ended by itself as the Escape went, its Stop heard already: stopped or ended, it runs no more.
            self._unowned.pop(prompt, None)
        turn.others += tuple(prompt for prompt in stopped if prompt not in turn.others)
        if stopped:
            logger.warning(f"the brain stopped prompts {list(stopped)}, which were no turn's")

    def interrupt(self) -> None:
        """Stop the turn in flight with Escape, as at the keyboard. Returns at once: the Escape is the brain's to press,
        and no hook says a turn was stopped, so the turn ends where it is pressed."""
        turn = self._turn
        if turn is not None:
            self._keep(self._stop(turn), turn)

    async def _stop(self, turn: _Turn) -> None:
        try:
            await self._escape(turn)
        except Exception as error:
            # A stop that breaks is never pressed again: its keys may have gone, and a second Ctrl-C within Claude Code's
            # exit window ends it. The turn ends with what broke it, and `_broke` says it.
            if turn.broken is None:
                turn.broken = error
            raise
        finally:
            # A broken turn ends with what broke it however its stop went, even unstopped: nothing else is left to end it,
            # and what it holds is refused first, so no dialog of it is left open for the next turn's Return. Unanswered,
            # it is still the turn in flight, so what is held is its own and no later turn's.
            if turn.broken is not None and not turn.answered.done():
                self._settle(Deny(BROKEN))
                self._over(turn, turn.broken)

    async def _escape(self, turn: _Turn) -> None:
        # [LAW:no-ambient-temporal-coupling] Escape goes once Claude Code has taken the turn, never before: until then
        # nothing of the turn's runs for it to stop.
        await asyncio.wait({turn.taken, turn.tried}, return_when=asyncio.FIRST_COMPLETED)
        if self._turn is not turn:
            return
        if not turn.taken.done():
            # [LAW:no-silent-failure] the turn runs on, told to stop by nobody: its words are the stage's to hold.
            logger.warning(f"the brain was not stopped: its turn was not taken, typed {turn.typings} time(s)")
            return
        try:
            async with self._input:
                if self._turn is not turn:
                    return
                await self._clear(turn)
        except BrainGone as error:
            self._over(turn, error)
            return
        if turn.answered.done():
            # Its Stop came while the Escape was pressed: the turn ended by itself, and says so once.
            return
        self._over(turn, BrainAnswered(turn.prompts[-1], None))

    def hear(self, observed: Observed) -> None:
        """The brain's own requests, read from the wire: the tools its turn offered the model, and whether they reached hands'."""
        match observed:
            case Sent(session=session, kind=MainTurn(), body=body) if session == self.session:
                tools = tool_names(body)
                if self._turn is not None:
                    self._turn.offered = tools
                # [LAW:no-silent-failure] a brain without hands' tools answers every question about the sessions from nothing,
                # and one with them only behind ToolSearch asks the model for them before it can answer.
                if not any(name.startswith(f"mcp__{SERVER_NAME}__") for name in tools):
                    logger.error(
                        f"the brain's turn went to the model without hands' tools and with {TOOL_SEARCH}: tool search is on"
                        f" though hands' --settings turn it off ({tools})"
                        if TOOL_SEARCH in tools
                        else f"the brain's turn went to the model without hands' tools: it did not connect to hands' MCP server ({tools})"
                    )
            case _:
                pass

    async def exited(self) -> int:
        """Waits for the brain to end, and returns its exit code; the line that says it ended is written once, however many wait."""
        return await asyncio.shield(self._exit)

    async def stop(self) -> None:
        await self._claude.stop()
        await self.exited()
        self._heard.cancel()
        await self._listener.cleanup()
        shutil.rmtree(self._sockets, ignore_errors=True)

    async def _type(self, typing: Callable[[Typist], None]) -> None:
        try:
            await asyncio.to_thread(typing, self._typist)
        except Untyped as error:
            raise BrainGone(f"the brain cannot be typed into: {error}") from error

    async def _run_out(self, up: Begun) -> int:
        # [LAW:nothing-unseen] the brain's run, from its input coming up to its process's end, is one event as it ends:
        # its exit code, and the last of what it showed on its terminal; or that hands stopped waiting for its end.
        with unit("brain.run", self._record, began=up):
            code = await asyncio.shield(self._claude.exit)
            annotate(pid=self.pid, code=code, shown=self._claude.shown())
            self._settle(Deny(f"the brain exited ({code})"))
            # [LAW:no-silent-failure] a turn that can never end is said to have failed, not left waiting.
            if self._turn is not None:
                self._over(self._turn, BrainGone(f"the brain exited ({code}) before it answered"))
        return code

    async def _hear_hooks(self, hooks: "asyncio.Queue[_Posted]") -> None:
        while True:
            posted = await hooks.get()
            # A hook is answered before it is heard, so one whose hearing breaks is no turn's: the hooks after it are heard.
            with self._done_for(None):
                self._answer(posted)

    def _answer(self, posted: _Posted) -> None:
        match posted.event:
            case "PermissionRequest":
                # What breaks in its settling is the turn's it was held for, which `_permit` finds; anything else is no turn's.
                self._keep(self._permit(posted), None)
            case "Elicitation":
                # A dialog is kept shut, whatever it says.
                posted.reply.set_result(DECLINED)
                self._refused(posted.body)
            case _:
                # Asks nothing of the turn, so it is answered at once, whatever it says.
                posted.reply.set_result({})
                self._hook(posted.body)

    async def _permit(self, posted: _Posted) -> None:
        """Answers a permission request: held while the user is asked, when a turn of theirs is in flight to ask in.

        [LAW:nothing-unseen] each is one event, however it was settled: a part of the turn it was held for, or of the
        brain's launch where no turn of the user's was in flight to ask in."""
        # What hands could not decide is refused, however its settling ended: a dialog left open would take the next
        # turn's Return for a yes.
        decision: Allow | Deny = Deny(BROKEN)
        try:
            try:
                said = Payload.parse(posted.body)
                prompt, tool, asked = said.optional_text("prompt_id"), said.text("tool_name"), called(said)
            except Rejected as error:
                decision = Deny(UNREAD)
                with unit("brain.permission", self._record):
                    fail(f"hands could not read the permission request: {error}")
                return
            # What breaks in its settling is its owner's: the turn whose user it was held to be put to.
            turn = self._owner(prompt)
            with self._done_for(turn), self._dialog("brain.permission", prompt, turn):
                annotate(tool=tool)
                try:
                    match asked:
                        case Permission() if turn is not None:
                            held = Asked(asked, asyncio.get_running_loop().create_future())
                            self._held.add(held)
                            try:
                                turn.asks(held)
                                await asyncio.wait({held.decision}, timeout=PERMISSION_DEADLINE_SECONDS)
                                held.settle(Deny(UNANSWERED))
                            finally:
                                self._held.discard(held)
                                # The user's side is told the refusal Claude Code is sent, and is not left waiting on it.
                                held.settle(Deny(BROKEN))
                            decision = held.decision.result()
                        case Permission():
                            decision = Deny(NOBODY)
                        case _:
                            # A dialog of questions or a plan is never put to the user by voice: the brain's own words ask them.
                            decision = Deny(UNVOICED)
                finally:
                    annotate(decision=decision)
        finally:
            posted.reply.set_result(hook_output(decision))

    @contextlib.contextmanager
    def _dialog(self, event: str, prompt: str | None, owner: _Turn | None) -> Generator[None]:
        """A dialog the brain posted, as one unit of work `event`: a part of its `owner`, the turn in flight it was posted
        for, or else of the brain's launch, for one a turn before it left behind or one posted between turns."""
        with continuing(self._launched if owner is None else owner.began.span), unit(event, self._record):
            annotate(prompt=prompt)
            yield

    def _owner(self, prompt: str | None) -> _Turn | None:
        """The turn in flight a dialog posted for `prompt` is a part of, if it is still in flight."""
        turn = self._turn
        # [LAW:single-enforcer] the turn's own, as its Stop is: one a turn before it left behind is nobody's to answer.
        return turn if turn is not None and not turn.answered.done() and prompt in turn.prompts else None

    def _settle(self, decision: Allow | Deny) -> None:
        """Settles every permission held now: its turn is over, so no answer of the user's can reach it."""
        for held in self._held:
            held.settle(decision)

    def _refused(self, body: bytes) -> None:
        """An MCP server's ask for input, answered no in a turn or between turns, where it has no prompt id: one event, so
        the refusal is not only the brain's to tell, a part of the turn it was posted in or else of the brain's launch."""
        try:
            said = Payload.parse(body)
            prompt, server = said.optional_text("prompt_id"), said.optional_text("mcp_server_name")
        except Rejected as error:
            with unit("brain.elicitation", self._record):
                fail(f"hands could not read the Elicitation hook: {error}")
            return
        with self._dialog("brain.elicitation", prompt, self._owner(prompt)):
            annotate(server=server)

    def _hook(self, body: bytes) -> None:
        try:
            said = Payload.parse(body)
            event, session = said.text("hook_event_name"), said.session_id()
            prompt = said.text("prompt_id")
            # What Claude Code took, on the hook that says it took a prompt: there it is never absent.
            submitted = said.text("prompt") if event == "UserPromptSubmit" else None
            failed = f"{said.optional_text('error')}: {said.optional_text('last_assistant_message')}"
        except Rejected as error:
            logger.warning(f"the brain posted a hook that does not parse: {error}")
            return
        if session != self.session:
            logger.info(f"the brain's {event} hook for prompt {prompt} came for session {session}, not the brain's")
            return
        turn = self._turn if self._turn is not None and not self._turn.answered.done() else None
        waiting = turn is not None and not turn.taken.done()
        match event, submitted:
            # [LAW:single-enforcer] a turn is taken by the prompt that is what it typed, never by the next to come: a turn
            # ended untaken may still be taken, and its prompt and its Stop are then no later turn's. One typed again is
            # taken by either typing, in whichever order their hooks come.
            case "UserPromptSubmit", str(words) if turn is not None and (waiting or turn.typings > 1) and _kept(words) == _kept(turn.typed):
                turn.prompts += (prompt,)
                if waiting:
                    turn.taken.set_result(prompt)
            case "UserPromptSubmit", _:
                # Runs as a whole turn of Claude Code's, ahead of what was typed behind it, until it ends or is stopped.
                self._unowned[prompt] = None
                if turn is not None and waiting:
                    turn.others += (prompt,)
                logger.warning(f"the brain took prompt {prompt}, which is no turn's: one typed for a turn that ended before it was taken, or a task's notification")
            case ("Stop" | "StopFailure"), _ if turn is not None and prompt in turn.prompts:
                self._over(turn, BrainAnswered(prompt, None if event == "Stop" else failed))
            case ("Stop" | "StopFailure"), _ if prompt in self._unowned:
                del self._unowned[prompt]
            case _ if turn is None:
                # A turn's own hook arriving after an Escape ended it, or a hook no turn of this brain's asked for.
                logger.info(f"the brain's {event} hook for prompt {prompt} came with no turn of its own in flight")
            case _:
                logger.warning(f"the brain's {event} hook for prompt {prompt} does not fit the turn in flight (prompts {turn.prompts})")

    def _over(self, turn: _Turn, outcome: BrainAnswered | BaseException) -> None:
        if self._turn is turn:
            self._turn = None
        if turn.answered.done():
            return
        # A broken turn ends with what broke it, whatever ended it: its Stop, its stop's Escape, or the brain's exit.
        match outcome if turn.broken is None else turn.broken:
            case BrainAnswered() as answered:
                turn.answered.set_result(answered)
            case failed:
                turn.answered.set_exception(failed)
                # Said already, by the turn's own event or by `_broke`: not again by asyncio for an asker that stopped waiting.
                turn.answered.exception()


async def start(launch: Launch, record: Record) -> Brain:
    """Start the brain under fritter on a terminal of hands' own, on the login its backend was parsed with."""
    station = launch.station
    # [LAW:nothing-unseen] the launch is one event, from the spawn until the brain's input is up, or what failed it.
    with unit("brain.launch", record):
        annotate(session=launch.session, account=launch.account, model=station.model, config_dir=station.config_dir, cwd=station.cwd)
        match launch.conversation:
            case Fresh(untranscribed=untranscribed):
                annotate(conversation="fresh", untranscribed=untranscribed)
            case Resumed():
                annotate(conversation="resumed")
        # [LAW:one-source-of-truth] the fritter built with this hands, never a copy installed apart from it: what hands
        # asks of fritter and what the brain's fritter answers are one build's.
        try:
            fritter = packaged()
        except Unpackaged as error:
            raise Unstartable(str(error)) from error
        annotate(fritter=fritter)
        claude = brain_claude(station.inherited)
        hooks: asyncio.Queue[_Posted] = asyncio.Queue()
        listener, url = await _listen(hooks)
        # A unix socket's path is capped near 104 bytes on macOS, so not under the brain's own directory.
        sockets = Path(tempfile.mkdtemp(prefix="hands-brain-"))
        running: ClaudeCode | None = None
        try:
            try:
                running = await spawn(station, [str(fritter), "--socket-dir", str(sockets), "--", *command(launch, claude, url)])
                annotate(pid=running.pid)
                typist = await _typist(running, sockets, launch.session)
            except Unstartable:
                # A conversation the brain could not come up in never fails the starts after it: the next begins anew, and
                # Claude Code keeps its transcript (2.1.289 exits "No conversation found" on one it cannot read).
                annotate(let_go=let_go(station.config_dir))
                raise
            # Held once the brain is up in it, so the next start resumes what this one says.
            hold(station.config_dir, launch.session)
        except BaseException:
            # [LAW:no-silent-failure] a start that fails or is cancelled leaves nothing running: the brain is in a session of
            # its own, which nothing but hands' own end would hang up.
            if running is not None:
                await running.stop()
            await listener.cleanup()
            shutil.rmtree(sockets, ignore_errors=True)
            raise
        launched = here()
    return Brain(running, launch.session, typist, hooks, listener, sockets, station.config_dir, record, launched)


async def _listen(hooks: "asyncio.Queue[_Posted]") -> tuple[web.AppRunner, str]:
    """The listener the brain posts its hooks to, on loopback, each hook put on `hooks` as it comes and answered with what
    the brain settles for it."""

    async def hook(request: web.Request) -> web.Response:
        # The event read off the path, so a body hands cannot read is still answered as its hook asks, by the brain,
        # whose record of the hook says it could not be read.
        reply: asyncio.Future[Mapping[str, object]] = asyncio.get_running_loop().create_future()
        hooks.put_nowait(_Posted(request.match_info["event"], await request.read(), reply))
        # Shielded: a handler aiohttp cancels leaves the reply to be set by the brain, which is never refused a hook.
        return web.json_response(await asyncio.shield(reply))

    app = web.Application()
    app.router.add_post("/{event}", hook)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def _typist(running: ClaudeCode, sockets: Path, session: SessionId) -> Typist:
    """The brain as fritter types into it, once its input is up: the socket fritter opened under `sockets`, and the claude
    it started.

    [LAW:no-ambient-temporal-coupling] up is the brain's own word, the bracketed paste Claude Code turns on once it reads
    its input (0.19s in, its terminal raw from 0.02s, 2.1.289, measured 2026-10-04), not a guess at how long that takes: a
    turn typed into a terminal not yet raw has its Return made a newline, which Claude Code takes as one more line."""
    deadline = asyncio.get_running_loop().time() + START_SECONDS
    waiting = "fritter had not started the brain and opened its socket"
    while asyncio.get_running_loop().time() < deadline and not running.exit.done():
        found = [path for path in sockets.rglob("*") if path.is_socket()]
        child = await asyncio.to_thread(_child_of, running.pid)
        if found and child is not None:
            typist = Typist(session, found[0], child)
            try:
                await asyncio.to_thread(typist.pasting)
                return typist
            except Untyped as error:
                waiting = f"the brain had not turned its input on ({error})"
        await asyncio.sleep(0.05)
    # What fritter's terminal showed says why: a fritter that could not be run at all fails there, in the shell it was run by.
    if running.exit.done():
        raise Unstartable(f"fritter exited ({running.exit.result()}) while {waiting}; it showed:\n{running.shown()}")
    raise Unstartable(f"{waiting} after {START_SECONDS:.0f}s; it showed:\n{running.shown()}")


def _child_of(pid: int) -> int | None:
    """The process `pid` started: fritter starts one, the claude it wraps."""
    found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return int(found[0]) if found else None
