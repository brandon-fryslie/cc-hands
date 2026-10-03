"""Whether a finished turn is told aloud: one switch, kept in the home, so it outlives the daemon and needs no restart.

The plugin's `/hands:summaries` skill runs this module under the plugin's own Python, which has only the standard
library, so it imports nothing else:

    hooks/python -m hands.sessions.summaries on|off

With no argument it says where the switch stands. The daemon reads the file at every finished turn and holds no copy
[LAW:one-source-of-truth], so a change is heard from the next turn on.
"""

import sys
from collections.abc import Sequence
from typing import Literal

from hands.sessions.files import replace_whole
from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected

Summaries = Literal["on", "off"]

# Off until the user turns them on: every finished turn of every session spoken was too much to sit through with
# several sessions (the user's decision, 2026-09-26).
DEFAULT: Summaries = "off"


def summaries(home: Home) -> Summaries:
    """Where the switch stands; the default where it was never set."""
    try:
        # Bytes, not text: a file edited by hand can hold anything, and whatever is not on or off is refused alike.
        written = home.summaries.read_bytes().strip()
    except FileNotFoundError:
        return DEFAULT
    match written:
        case b"on":
            return "on"
        case b"off":
            return "off"
        case _:
            raise Rejected(f"{home.summaries} says {written!r}, which is neither on nor off")


def set_summaries(home: Home, to: Summaries) -> None:
    # [LAW:no-ambient-temporal-coupling] replaced whole, so the daemon never reads half a file.
    replace_whole(home.summaries, f"{to}\n", 0o644)


def described(to: Summaries) -> str:
    match to:
        case "on":
            return "Spoken turn summaries are on: every turn a session finishes is told aloud."
        case "off":
            return "Spoken turn summaries are off: only a watched session's turns are told as they finish, and any session's last turn when you ask for it."


def main(argv: Sequence[str]) -> int:
    try:
        home = default_home()
        # [LAW:parse-dont-validate] read as a person types it: "On" is on.
        match [argument.lower() for argument in argv]:
            case []:
                print(described(summaries(home)))
            case ["on" | "off" as to]:
                set_summaries(home, to)
                print(described(to))
            case _:
                print(f"hands summaries: expected on or off, got {' '.join(argv)!r}", file=sys.stderr)
                return 2
    except (Rejected, OSError) as error:
        print(f"hands summaries: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
