"""Whisper's vocabulary for a hold: the words the user is likely to say that no dictionary holds, read as the hold is
transcribed, so a focus moved a moment ago primes the next thing said [LAW:one-source-of-truth].

Whisper takes text as its initial prompt, read as what was said before the audio, and spells what it hears the way
that text does. LowTalker reads the prompt it is sent as one vocabulary term and refuses one past the 111 prompt tokens
its engine keeps, so the words are fitted to that before they are sent. Ten sentences spoken by `say`, each naming one of hands' own identifiers, came back from
large-v3-turbo with none of the ten spelled as named unprimed, and six primed with the ten; "auth middleware", said in
ten sentences, came back as authMiddleware eight times primed with what this module read from a repository holding
authMiddleware.ts among thirty files, and never unprimed (2026-10-03). The words are the files most recently changed in
the focused session's repository and its branch, which the user says while working in it, then the projects and names
of the running sessions, which they say to move between them.
"""

import asyncio
import base64
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import tiktoken

from hands.sessions.audit import Primed, Record
from hands.sessions.child import run
from hands.sessions.focus import Unreadable, focused
from hands.sessions.home import Home
from hands.sessions.registry import Listing, Sessions
from hands.core.session import Session
from hands.voice.readback import identifier

# The most words Whisper is primed with. The same ten sentences, their ten identifiers placed among the names of the
# files hands changed most recently, came back with six spelled as named among 40 words, three among 70 and four among
# 130: past a few dozen, the words the user said are drowned by the ones they did not.
WORDS = 40

# The most prompt tokens LowTalker's engine keeps (WhisperKit's), past which it refuses the prompt.
TOKENS = 111

# The commits whose files are read, newest first: as far back as the work a session is in is likely to reach.
COMMITS = 30

# What reading the vocabulary may spend, the sessions' names and the repository's commands together. Whisper waits on it
# before it transcribes, so it is time added to every turn; where git is still running past it, Whisper is primed with
# the sessions' names alone.
READING = 1.0


class Lexicon:
    """Reads Whisper's vocabulary as each hold is transcribed, and records what it read."""

    def __init__(self, sessions: Sessions, home: Home, environment: Mapping[str, str], record: Record) -> None:
        self._sessions = sessions
        self._home = home
        # The daemon's environment, which git is found and run in.
        self._environment = environment
        self._record = record

    async def __call__(self) -> str | None:
        began = time.monotonic()
        # Read off the loop: a name is read from its session's transcript.
        listings = await asyncio.to_thread(self._sessions.live)
        primed = await vocabulary(listings, self._focus(), self._environment, began)
        self._record(primed)
        # [LAW:dataflow-not-control-flow] no words is no prompt, which is Whisper unprimed.
        return prompt(primed.words) or None

    def _focus(self) -> Session | Unreadable | None:
        match focused(self._home):
            case Unreadable() | None as unfocused:
                return unfocused
            case session:
                # A focused session that has ended primes nothing of its own, as one never focused does not.
                return self._sessions.live_session(session)


async def vocabulary(listings: Sequence[Listing[Session]], focus: Session | Unreadable | None, environment: Mapping[str, str], began: float) -> Primed:
    """The repository words of the focused session, then every running session's project and name, oldest first, the
    oldest dropped until the rest fit the prompt tokens Whisper keeps. `began` is when reading the vocabulary started,
    `listings` among it, on the monotonic clock."""
    match focus:
        case Session() as session:
            repository, failed = await _repository(session.membership.cwd, environment, began + READING)
            focus_id = session.membership.id
        case Unreadable(reason=reason):
            repository, failed, focus_id = (), f"the focus: {reason}", None
        case None:
            repository, failed, focus_id = (), None, None
    # [LAW:one-source-of-truth] each session as it is spoken and addressed.
    sessions = tuple(identifier(listing) for listing in listings)
    words = _unique([word for word in (*repository, *sessions) if not _SPECIAL.search(word)])[-WORDS:]
    while (tokens := _tokens(words)) > TOKENS:
        words = words[1:]
    return Primed(focus_id, words, tokens, failed, time.monotonic() - began)


def prompt(words: Sequence[str]) -> str:
    """The prompt `words` are sent as: space-joined, so no punctuation of the prompt's own is read back into what was heard."""
    return " ".join(words)


def _tokens(words: Sequence[str]) -> int:
    """The prompt tokens LowTalker counts for the prompt of `words`: the one term it reads it as, with the leading space a
    spoken word carries, in Whisper's own BPE; none for none. Measured against the tokenizer LowTalker loads, 2000 prompts
    of up to 40 file, branch and session names counted the same (2026-10-04)."""
    return len(_WHISPER_BPE.encode_ordinary("".join(f" {word}" for word in words)))


def _whisper_bpe() -> tiktoken.Encoding:
    """Whisper's multilingual BPE (whisper.tiktoken, from openai/whisper): each line a base64 token and its rank."""
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in (Path(__file__).parent / "whisper.tiktoken").read_text().splitlines() if line)}
    # Whisper's pre-tokenizer, GPT-2's.
    return tiktoken.Encoding("whisper", pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", mergeable_ranks=ranks, special_tokens={})


# Read as the module is imported, with Pipecat while hands starts, so no hold waits the ~50 ms reading it takes.
_WHISPER_BPE = _whisper_bpe()
# A special token's text, as Whisper's are all spelled: one in a prompt is read as that token, so a file named
# <|endoftext|> would end every prompt it is in.
_SPECIAL = re.compile(r"<\|[^|<>]*\|>")


async def _repository(cwd: Path, environment: Mapping[str, str], deadline: float) -> tuple[tuple[str, ...], str | None]:
    """The names of the files most recently changed in the repository `cwd` is in, oldest first, then its branch; or
    none, and why, where git could not say. A directory in no repository has none, and that is no failure."""
    try:
        logged, changed, branch = await asyncio.gather(
            # A branch with no commits yet has no history to read, where git log would refuse its unborn HEAD.
            _git(cwd, environment, deadline, "log", "--ignore-missing", "HEAD", "--name-only", "--format=", "-z", f"-n{COMMITS}"),
            _git(cwd, environment, deadline, "status", "--porcelain=v1", "-z"),
            _git(cwd, environment, deadline, "branch", "--show-current"),
        )
    except NotARepository:
        return (), None
    except GitFailed as error:
        return (), str(error)
    # git log lists the newest commit first, and git status what is not committed yet, which is newer still.
    files = (*reversed(logged.split("\0")), *_status_paths(changed))
    return (*(Path(path).stem for path in files if path), *branch.split()), None


def _status_paths(status: str) -> list[str]:
    """The paths `git status --porcelain=v1 -z` names, without the source of a rename or copy."""
    entries = iter(status.split("\0"))
    paths: list[str] = []
    for entry in entries:
        if entry:
            paths.append(entry[3:])
            if "R" in entry[:2] or "C" in entry[:2]:
                # The source follows its destination as an entry of its own, and was not changed in this work.
                next(entries)
    return paths


class GitFailed(Exception):
    """git could not say what the repository holds."""


class NotARepository(GitFailed):
    """The directory is in no git repository."""


async def _git(cwd: Path, environment: Mapping[str, str], deadline: float, *args: str) -> str:
    left = deadline - time.monotonic()
    if left <= 0:
        raise GitFailed(f"reading the sessions' names spent the vocabulary's {READING:.1f}s before git {args[0]} in {cwd} could run")
    try:
        # git says why it failed in the C locale's words, which are the ones read below.
        ran = await run("git", "--no-optional-locks", "-C", str(cwd), *args, timeout=left, env={**environment, "LC_ALL": "C"})
    except TimeoutError:
        raise GitFailed(f"git {args[0]} in {cwd} was still running when reading the vocabulary had spent its {READING:.1f}s") from None
    except OSError as error:
        raise GitFailed(f"cannot run git in {cwd}: {error}") from None
    said = ran.err.decode(errors="replace").strip()
    # [LAW:domain-language] git's own words for a directory outside every repository, and the exit code it gives them.
    if ran.returncode == 128 and "not a git repository" in said:
        raise NotARepository(said)
    if ran.returncode != 0:
        raise GitFailed(f"git {args[0]} in {cwd} exited {ran.returncode}: {said}")
    return ran.out.decode(errors="replace")


def _unique(words: Sequence[str]) -> tuple[str, ...]:
    """Each word once, where it was last: the newest mention of a file is the one that says how recent it is."""
    return tuple(reversed(dict.fromkeys(reversed(words))))

