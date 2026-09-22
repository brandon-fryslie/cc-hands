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
_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_DIFF_PREAMBLE = re.compile(r"^(index [0-9a-f]|new file|deleted file|similarity index|rename |old mode|new mode|--- |\+\+\+ |Binary files )")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
# Two digits at most, because a numbered list counts and a year does not: "2024. It was a good year."
# is a sentence, and read as a list marker it lost the year off the front of itself.
_BULLET = re.compile(r"^[ \t]*(?:[-*+]|\d{1,2}[.)])[ \t]+(.*)$")
_CONTINUED = re.compile(r"^[ \t]+\S")
_CODE = re.compile(r"`+([^`\n]+)`+")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
# Stopping at the first `)` left the rest of the address behind, and a stray bracket was read aloud.
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\((?:[^()\s]|\([^()\s]*\))*\)")
_AUTOLINK = re.compile(r"<(?:https?://|www\.)[^\s>]*>")
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
# Both cases, not merely one: `HTTP_TIMEOUT_30` has an upper and `update_database_2` has a lower, and a
# config constant named out loud in a summary is a fact, not an id to be dropped whole.
# And length is the tell that separates an id from a name, because every other one they share.
# `Base64Decoder`, `Float32Array`, `HTTP2Handler`, `Sha256Checksum` and `OAuth2TokenStore` are long,
# numbered and mixed case, and are names `_CAMEL` exists to say; they measure 12 to 16 characters.
# `011CewaSjsqPyMa9o6U1HSUg` and `toolu_01ShPUZU4n3cWZ5ikycdXMZj` measure 24 to 29. A name is as long as
# a person cared to type and an id is as long as the machine that made it needed, so the gap is the line.
def _id(least: int) -> str:
    return rf"(?=[A-Za-z0-9_]{{{least},}}\b)(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Z])(?=[A-Za-z0-9_]*[a-z])[A-Za-z0-9_]+\b"


_OPAQUE = re.compile(rf"\b{_id(20)}")
# Shorter where the sentence has already said what it is: the word in front of it is the tell, so the
# length does not have to be. Nouns only — "run" is a verb, and as a trigger it deleted the object of
# every sentence it stood in front of. And the trigger word is what ignores case here, nothing else:
# setting the flag on the whole pattern reached inside the tell and cancelled it, so the same name was
# dropped after "token" and spoken anywhere else.
_NAMED_OPAQUE = re.compile(rf"\b((?i:ids?|tokens?|requests?|sessions?))\s+{_id(12)}")
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


# A ref's separators, none of which can be heard: "feature/narration-tree" is four words and three noises.
_REF_SEPARATORS = str.maketrans("/-_", "   ")


def spoken_ref(ref: str) -> str:
    """A git ref said out loud, for a caller whose type already knows it is one.

    `_PATH` deliberately will not read `feature/narration-tree` as a path: a rule loose enough to catch it also
    catches "and/or", "24/7" and "input/output", and costs every one of them a word. So a bare ref is recognised
    only where something already knows what it is — a `Pushed` or a `Branched`, never a guess made from prose —
    and saying it is then just a matter of the separators [LAW:single-enforcer]. Every word is kept: a branch is
    named so it can be told from the others, and "release/1.2" heard as "1.2" is the one thing it is not.
    """
    return " ".join(ref.translate(_REF_SEPARATORS).split())


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


def _leaked(kind: str, lines: list[str], leaks: list[Leak]) -> list[str]:
    """What is said in place of the block, or nothing where the block held nothing.

    An empty fence and a table of nothing but its own rule have no content to apologise for. Saying "a
    block of code of 0 lines" tells the listener something was there when nothing was — and warning about
    it [LAW:no-silent-failure] would report a fault every time the closing fence of a block split across
    two streamed chunks arrived on its own, which is the shape this seam is already known to produce.
    """
    if not lines:
        return []
    leak = Leak(kind, len(lines))
    leaks.append(leak)
    return [f"{leak}."]


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
            out.extend(_leaked("code", held, leaks))
            held = None
            continue
        (out if held is None else held).append(line)
    if held is not None:
        out.extend(_leaked("code", held, leaks))
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
    return _leaked("table", [line for line in run if not set(line) <= set("|-: \t")], leaks)


def _undiffed(text: str, leaks: list[Leak]) -> str:
    """A diff, which only counts as one where it says so, and runs exactly as far as it says it does.

    Prose starts with a dash often enough that a leading `-` is no tell at all, so a diff must announce
    itself. And a hunk header states how many lines it covers on each side, so the end of the diff is read
    off the diff rather than guessed from the shape of the next line [LAW:parse-dont-validate]. Guessed,
    a bulleted list written straight under a hunk was swallowed whole — every line of it opens with a
    dash, which is also how a removal opens — and the listener heard a line count instead of the reply.
    """
    lines = text.splitlines()
    out: list[str] = []
    at = 0
    while at < len(lines):
        if not _DIFF_OPENS.match(lines[at]):
            out.append(lines[at])
            at += 1
            continue
        ends = _diff_ends(lines, at)
        out.extend(_leaked("diff", lines[at:ends], leaks))
        at = ends
    return "\n".join(out)


def _diff_ends(lines: list[str], start: int) -> int:
    """The line the diff beginning at `start` stops before, by its own account."""
    at = start
    while at < len(lines):
        hunk = _HUNK.match(lines[at])
        if hunk:
            # A header with no count covers one line of each side, which is what `@@ -1 +1 @@` means.
            at = _hunk_ends(lines, at + 1, int(hunk.group(1) or 1), int(hunk.group(2) or 1))
        elif at == start or lines[at].startswith("diff --git ") or _DIFF_PREAMBLE.match(lines[at]):
            at += 1
        else:
            break
    return at


def _hunk_ends(lines: list[str], at: int, old: int, new: int) -> int:
    """The line the hunk body stops before: where both sides have had every line they were promised.

    A line only counts as what it looks like while the side it belongs to still has room. That is what
    ends a hunk at a blank line the diff never claimed, rather than counting the blank that separates the
    patch from the sentence after it — and the count is the whole of what the listener is told about a
    block they will never hear [LAW:no-silent-failure].
    """
    while at < len(lines) and (old or new):
        line = lines[at]
        if line.startswith("\\"):  # "\ No newline at end of file" belongs to neither side.
            pass
        elif line.startswith("-") and old:
            old -= 1
        elif line.startswith("+") and new:
            new -= 1
        elif old and new and (not line or line.startswith(" ")):
            old, new = old - 1, new - 1
        else:
            break
        at += 1
    return at


def _headings(text: str) -> str:
    """A heading is the cue that a new part has started, and in speech a cue is a sentence of its own."""
    return _HEADING.sub(lambda found: f"{found.group(1).rstrip('.:')}.", text)


def _lists(text: str) -> str:
    """A list read straight through is heard as one long sentence, so the items are counted aloud."""
    out: list[str] = []
    run: list[str] = []
    # A blank line between items is how a model writes a list as often as not, and a run that ended on one
    # became a string of single items — which `_counted` then declines to number, so the one shape this
    # function most exists for was the one shape it did nothing to.
    held: list[str] = []
    for line in text.splitlines():
        found = _BULLET.match(line)
        if found:
            run.append(found.group(1))
            held = []
            continue
        if run and not line.strip():
            held.append(line)
            continue
        if run and not held and _CONTINUED.match(line):
            # An item wrapped onto the next line is still that item. Counted as the end of the run, the
            # item after it was announced to the listener as the first [LAW:no-ambient-temporal-coupling].
            run[-1] = f"{run[-1]} {line.strip()}"
            continue
        out.extend(_counted(run))
        run = []
        out.extend(held)
        held = []
        out.append(line)
    out.extend(_counted(run))
    out.extend(held)
    return "\n".join(out)


def _counted(run: list[str]) -> list[str]:
    if not run:
        return []
    if len(run) == 1:
        # One bullet is not a sequence, and "First," in front of a lone item says there is a second.
        return [_stopped(run[0])]
    return [" ".join(f"{_ordinal(place)}, {_stopped(item)}" for place, item in enumerate(run))]


def _stopped(item: str) -> str:
    """An item ends in a full stop unless it already ends in something that stops it.

    Added unconditionally, a bulleted question became "Should it fix them?." and the pair reached the
    speaker [LAW:one-source-of-truth]: one rule for how an item ends, wherever the item came from.
    """
    return item if item.endswith((".", "?", "!")) else f"{item}."


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
    # The brackets of an address written inside them go with it, rather than being left for `_tidied` to
    # sweep up: `<` and `>` mean less than and greater than in every other sentence they appear in.
    text = _AUTOLINK.sub("a link", text)
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
    # Matched as written rather than lowered: `Deno.Go` is a sentence that lost its space, not a Go file,
    # and lowering it deleted the word after the full stop [LAW:carrying-cost].
    return _FILE.sub(lambda found: found.group(1) if found.group(2) in _EXTENSIONS else found.group(0), text)


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
    # Four of these marks say something in an ordinary sentence, so each is dropped only in the shape that
    # makes it markdown [LAW:carrying-cost]. Swept up unconditionally, "Latency is now < 200 ms" became
    # "Latency is now 200 ms" — grammatical, and a different fact than the one that was written.
    text = re.sub(r"^[ \t]*>+[ \t]*", "", text, flags=re.MULTILINE)  # a blockquote marker, which is drawn
    text = text.replace("~~", " ")  # struck-through text, which is doubled where "about 500" is not
    text = re.sub(r"\*+(?=\S)|(?<=\S)\*+", " ", text)  # emphasis left unpaired, where `3 * 4` is spaced
    text = re.sub(r"[`|_#]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # A rule across the page and the dashes under a heading are drawn, not written: they are a line with
    # nothing in it but marks. Dropped whole rather than by character, because a dash inside a line is a
    # hyphen and a minus sign, and speech wants both of those kept.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip() and not set(line.strip()) <= set("-=+. ")).strip()
