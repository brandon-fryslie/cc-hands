"""What can be heard: everything on its way to the speaker passes through here, and nothing else does.

Markdown, code, diffs, tables, paths, hashes and URLs cannot be heard as written. A speech engine reads
`user_id` as "user underscore id", `src/auth.py` as a string of punctuation, and a fenced block as a minute
of syllables nobody can follow. Asking a model to write for the ear does not settle it: that is a rule held
as an instruction, obeyed or not, checked by nobody — and on this machine the summariser has already been
heard saying a bare file name, while session titles never passed the instruction at all. So spoken form is
enforced where it can be, at the one seam every utterance crosses on its way out [LAW:single-enforcer].

Pure, and stdlib only, because it is the domain: what a developer who is not looking can hear. A leak is
returned rather than logged, because logging is an effect and this is not the edge [LAW:effects-at-boundaries].
"""

import re
from dataclasses import dataclass

# What a leaked block is said as, instead of read out. Kept short: it is an apology, not the content.
_KINDS = {"code": ("a block of code", "line"), "diff": ("a diff", "line"), "table": ("a table", "row")}

# Counted sequences, because a list read as a run-on sentence is heard as one thing.
_ORDINALS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth")

# Each rule asks for a tell that ordinary English does not have, because a rule that mangles a sentence
# costs more than the code name it fixes [LAW:carrying-cost]. That is why a diff must announce itself, a
# table must be two rows rather than one line with a pipe in it, and a hash must carry a digit and a letter
# both. A rule below with no tell is a bug in this module's terms, and `test_spoken.py` holds it to them.
_FENCE = re.compile(r"^[ \t]*(?P<run>`{3,}|~{3,})(?P<info>.*)$")
_DIFF_OPENS = re.compile(r"^(diff --git |@@ )")
_DIFF_BODY = re.compile(r"^([+\-@\\ ]|index [0-9a-f]|new file|deleted file|similarity index)")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+(.*)$")
_CODE = re.compile(r"`+([^`\n]+)`+")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\n]+\)")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>]*[^\s<>.,;:!?)\]'\"]")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
# Both tells at once, and neither alone is one. Letters without a digit is a word — "defaced" is a perfectly
# good past participle. Digits without a letter is a number, and a file of 1048576 bytes is a fact a
# developer who is not looking actually needs [LAW:carrying-cost].
_HEX = r"(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]+\b"
_SHA = re.compile(rf"\b{_HEX}", re.IGNORECASE)
_NAMED_SHA = re.compile(rf"\b(commits?|sha|hash|revision|rev)\s+{_HEX}", re.IGNORECASE)
# An id is mixed case as well as long and numbered, because `base64_encode` is all three of long, numbered
# and made of words — and a name made of words is `_SNAKE`'s to say, not an id to be dropped whole.
_ID = r"(?=[A-Za-z0-9_]{12,}\b)(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Z])[A-Za-z0-9_]+\b"
_OPAQUE = re.compile(rf"\b{_ID}")
_NAMED_OPAQUE = re.compile(rf"\b(ids?|tokens?|requests?|sessions?|runs?)\s+{_ID}", re.IGNORECASE)
_FILE = re.compile(r"\b([\w\-]+)\.([A-Za-z0-9]{1,5})\b")

# A closed list, so no ordinary sentence is ever mistaken for a file name. "e.g." and "etc." and a domain
# are all `word.word`, and a rule that guessed would cost more than the file names it caught
# [LAW:carrying-cost]. A new extension is one entry here.
_EXTENSIONS = frozenset(
    "py js ts tsx jsx md txt json jsonl toml yaml yml sh bash zsh rs go java rb c h cpp cc css html lock cfg ini sql csv xml png jpg svg".split()
)
# A path says so the way prose never does: it starts at the root with a directory above it, or its last
# word ends in one of the extensions above [LAW:one-source-of-truth]. Asking for neither made "and/or",
# "input/output", "24/7", "12/25/2025" and "1/2" all paths, and every one of them lost a word.
_PATH = re.compile(
    r"(?<![\w/])(?:/(?:[\w.\-]+/)+[\w.\-]+"
    rf"|(?:[\w.\-]+/)+[\w\-]+\.(?:{'|'.join(sorted(_EXTENSIONS))}))\b"
)
_FLAG = re.compile(r"(?<![\w-])--?([A-Za-z][\w-]*)(=)?")
_DOTTED_CALL = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b(?=\s*\()")
# Three names joined by dots is a module; two is "e.g." or a sentence that ended without a space after it.
_DOTTED_NAME = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}\b")
_SNAKE = re.compile(r"\b\w*_\w*\b")
_CAMEL = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b")


@dataclass(frozen=True)
class Leak:
    """Something speech cannot carry at all, which reached the ear instead of the summariser.

    Said as what it was and how big it was, because a developer who cannot see the screen still needs to
    know something was there. The daemon logs it: a leak means a summary that should have been made was
    not, and that is a fault upstream rather than here [LAW:no-silent-failure].
    """

    kind: str
    lines: int

    def __str__(self) -> str:
        named, unit = _KINDS.get(self.kind, (self.kind, "line"))
        return f"{named} of {self.lines} {unit}{'' if self.lines == 1 else 's'}"


@dataclass(frozen=True)
class Spoken:
    """Text that can be heard, and whatever had to be taken out of it whole to make it so."""

    text: str
    leaks: tuple[Leak, ...] = ()


def spoken(text: str) -> Spoken:
    """`text` in a form that can be spoken: the one conversion every utterance goes through.

    The rules run in an order, and the order is the design. What speech cannot carry at all comes out
    first, whole, so that no later rule reads a path out of a diff or an identifier out of a table; then
    the shape of the writing becomes the shape of the speech; then what is left of each word is made
    sayable. A rule that would damage an ordinary English sentence is not worth the code name it fixes,
    so each one below asks for a tell that prose does not have [LAW:carrying-cost].
    """
    leaks: list[Leak] = []
    said = _unfenced(text, leaks)
    said = _untabled(said, leaks)
    said = _undiffed(said, leaks)
    said = _headings(said)
    said = _lists(said)
    said = _emphasis(said)
    said = _links(said)
    said = _hashes(said)
    said = _paths(said)
    said = _flags(said)
    said = _identifiers(said)
    return Spoken(_tidied(said), tuple(leaks))


def _leaked(kind: str, lines: list[str], leaks: list[Leak]) -> str:
    leak = Leak(kind, len(lines))
    leaks.append(leak)
    return f"{leak}."


def _unfenced(text: str, leaks: list[Leak]) -> str:
    """A fenced block, said as what it was and how long it was.

    An unclosed fence runs to the end of the text, because that is what a reply cut off mid-block leaves,
    and reading the rest of it out is the one thing this exists to prevent.
    """
    out: list[str] = []
    held: list[str] | None = None
    fence = ""
    for line in text.splitlines():
        found = _FENCE.match(line)
        if held is None:
            # A backtick fence may not carry a backtick in what follows it, so "```bash``` is what I ran"
            # opens nothing: it is a sentence with an inline span at the front of it, and treating it as a
            # fence swallowed every line after it to the end of the reply [LAW:parse-dont-validate].
            if found and not (found["run"][0] == "`" and "`" in found["info"]):
                held, fence = [], found["run"]
                continue
        elif found and found["run"][0] == fence[0] and len(found["run"]) >= len(fence) and not found["info"].strip():
            # Closed only by its own fence, at least as long. A four-backtick block is how a model quotes a
            # three-backtick one, and a closer that ignored length ended the outer block at the inner
            # opening — which read the quoted code out loud, the one thing this exists to prevent.
            out.append(_leaked("code", held, leaks))
            held = None
            continue
        (out if held is None else held).append(line)
    if held is not None:
        out.append(_leaked("code", held, leaks))
    return "\n".join(out)


def _untabled(text: str, leaks: list[Leak]) -> str:
    """A pipe table. One row is a sentence with a pipe in it; two in a row is a table nobody can hear."""
    out: list[str] = []
    run: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            run.append(line)
            continue
        out.extend(_table(run, leaks))
        run = []
        out.append(line)
    out.extend(_table(run, leaks))
    return "\n".join(out)


def _table(run: list[str], leaks: list[Leak]) -> list[str]:
    if len(run) < 2:
        return run
    # The rule under a table's heading row is drawn, not said, so it is not one of the rows counted.
    return [_leaked("table", [line for line in run if not set(line) <= set("|-: \t")], leaks)]


def _undiffed(text: str, leaks: list[Leak]) -> str:
    """A diff, which only counts as one where it says so: prose starts with a dash often enough that a
    leading `-` is no tell at all, and a list of three points would be swallowed as a patch."""
    out: list[str] = []
    held: list[str] | None = None
    # A blank line inside a diff is part of it; the same blank line is part of the text again if the diff
    # turns out to have ended there. Held aside until the next line says which, because the count is the
    # only thing the listener is given about a block they will never hear [LAW:no-silent-failure].
    blank: list[str] = []
    for line in text.splitlines():
        if held is None and _DIFF_OPENS.match(line):
            held = [line]
            continue
        if held is not None:
            if not line.strip():
                blank.append(line)
                continue
            if _DIFF_BODY.match(line):
                held.extend(blank)
                held.append(line)
                blank = []
                continue
            out.append(_leaked("diff", held, leaks))
            held = None
            out.extend(blank)
            blank = []
        out.append(line)
    if held is not None:
        out.append(_leaked("diff", held, leaks))
        out.extend(blank)
    return "\n".join(out)


def _headings(text: str) -> str:
    """A heading is the cue that a new part has started, and in speech a cue is a sentence of its own."""
    return _HEADING.sub(lambda found: f"{found.group(1).rstrip('.:')}.", text)


def _lists(text: str) -> str:
    """A list read straight through is heard as one long sentence, so the items are counted aloud."""
    out: list[str] = []
    run: list[str] = []
    for line in text.splitlines():
        found = _BULLET.match(line)
        if found:
            run.append(found.group(1))
            continue
        out.extend(_counted(run))
        run = []
        out.append(line)
    out.extend(_counted(run))
    return "\n".join(out)


def _counted(run: list[str]) -> list[str]:
    if not run:
        return []
    if len(run) == 1:
        # One bullet is not a sequence, and "First," in front of a lone item says there is a second.
        return [run[0] if run[0].endswith((".", "?", "!")) else f"{run[0]}."]
    return [" ".join(f"{_ordinal(place)}, {item.rstrip('.')}." for place, item in enumerate(run))]


def _ordinal(place: int) -> str:
    return _ORDINALS[place] if place < len(_ORDINALS) else f"Number {place + 1}"


def _emphasis(text: str) -> str:
    """Inline code is the words inside it; bold and italic are nothing at all, being marks for the eye."""
    text = _CODE.sub(lambda found: found.group(1), text)
    text = _BOLD.sub(lambda found: found.group(1) or found.group(2), text)
    return _ITALIC.sub(lambda found: found.group(1), text)


def _links(text: str) -> str:
    """A written link already says what it points at; an address says nothing that can be heard."""
    text = _MD_LINK.sub(lambda found: found.group(1), text)
    return _URL.sub("a link", text)


def _hashes(text: str) -> str:
    """Named by what they are, since no digit of either is worth hearing and both are unrepeatable aloud.

    A sha must carry a digit to count as one, because seven letters drawn from a to f is also a word:
    "defaced" is a perfectly good English past participle and a perfectly good short hash. Where the
    sentence already says which kind of thing it is, the name is the whole of what is left: "commit
    a1b2c3d" is heard as "commit", not as "commit a commit".
    """
    text = _NAMED_SHA.sub(lambda found: found.group(1), text)
    text = _SHA.sub("a commit", _UUID.sub("an id", text))
    # A dozen characters of mixed letters and digits is a request id, a token, or a run name — something
    # that was only ever meant to be copied, and that speech can do nothing with but spell out.
    text = _NAMED_OPAQUE.sub(lambda found: found.group(1), text)
    return _OPAQUE.sub("an id", text)


def _paths(text: str) -> str:
    """A path is heard as its file, which is what a developer calls it out loud.

    The directory comes back only where two different paths in the same breath would otherwise be the
    same word — which is the whole reason a directory is ever said, and no reason to say it otherwise.
    """
    paths = _PATH.findall(text)
    shared = {_stem(path) for path in paths if any(other != path and _stem(other) == _stem(path) for other in paths)}
    text = _PATH.sub(lambda found: _spoken_path(found.group(0), shared), text)
    # A bare file name is a path with nothing in front of it, and is what the summariser was actually
    # heard saying out loud: "notes dot text" is not how anybody refers to the file they just changed.
    return _FILE.sub(lambda found: found.group(1) if found.group(2).lower() in _EXTENSIONS else found.group(0), text)


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _spoken_path(path: str, shared: set[str]) -> str:
    parts = path.split("/")
    if len(parts) < 2 or _stem(path) not in shared:
        return _stem(path)
    return f"{parts[-2]} {_stem(path)}"


def _flags(text: str) -> str:
    """A flag is its name: the dashes are how it is typed, not what it is called."""
    # The `=` of `--max-count=5` is typing too: what is said is the name and then the value.
    return _FLAG.sub(lambda found: found.group(1).replace("-", " ") + (" " if found.group(2) else ""), text)


def _identifiers(text: str) -> str:
    """The symbols in a code name are read aloud one by one, so the name becomes the words it is made of."""
    text = _DOTTED_CALL.sub(lambda found: found.group(0).replace(".", " "), text)
    text = _DOTTED_NAME.sub(lambda found: found.group(0).replace(".", " "), text)
    text = _SNAKE.sub(lambda found: found.group(0).replace("_", " ").strip(), text)
    return _CAMEL.sub(lambda found: re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", found.group(0)).lower(), text)


def _tidied(text: str) -> str:
    """Whatever the rules left behind. Speech has no use for a mark, so the ones that survived are dropped.

    Last rather than first, and unconditional: it is what makes "no backtick, no pipe, no fence reaches
    the speaker" a property of this function rather than a hope about the rules above it.
    """
    # Empty parentheses are how a function is written, never how one is named out loud.
    text = text.replace("()", " ")
    text = re.sub(r"[`|*_#<>~]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # A rule across the page and the dashes under a heading are drawn, not written: they are a line with
    # nothing in it but marks. Dropped whole rather than by character, because a dash inside a line is a
    # hyphen and a minus sign, and speech wants both of those kept.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip() and not set(line.strip()) <= set("-=+. ")).strip()
