"""Whether a finished turn is told aloud: one switch, kept in the home, so it outlives the daemon and needs no restart.

The plugin's `/hands:summaries` skill runs this module under the plugin's own Python, which has only the standard
library, so it imports nothing else:

    hooks/python -m hands.sessions.summaries on|off

With no argument it says where the switch stands. The daemon reads the file at every finished turn and holds no copy
[LAW:one-source-of-truth], so a change is heard from the next turn on.
"""

import os
import sys
from collections.abc import Sequence
from typing import Literal

from hands.sessions.home import Home, default_home
from hands.sessions.payload import Rejected

Summaries = Literal["on", "off"]

# Off until the user turns them on: every finished turn of every session spoken was too much to sit through with
# several sessions (the user's decision, 2026-09-26).
DEFAULT: Summaries = "off"


def summaries(home: Home) -> Summaries:
    """Where the switch stands; the default where it was never set."""
    try:
        written = home.summaries.read_text().strip()
    except FileNotFoundError:
        return DEFAULT
    match written:
        case "on" | "off":
            return written
        case _:
            raise Rejected(f"{home.summaries} says {written!r}, which is neither on nor off")


def set_summaries(home: Home, to: Summaries) -> None:
    home.root.mkdir(parents=True, exist_ok=True)
    # [LAW:no-ambient-temporal-coupling] written beside and renamed into place, so the daemon never reads half a file.
    staging = home.summaries.with_suffix(f".{os.getpid()}.tmp")
    staging.write_text(f"{to}\n")
    staging.replace(home.summaries)


def described(to: Summaries) -> str:
    match to:
        case "on":
            return "Spoken turn summaries are on: every turn a session finishes is told aloud."
        case "off":
            return "Spoken turn summaries are off: a finished turn is said only when it asks you something."


def main(argv: Sequence[str]) -> int:
    try:
        home = default_home()
        match argv:
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
