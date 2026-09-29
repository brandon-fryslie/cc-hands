"""The wall clock Claude Code stamps its statuses and records with, read on this side in the same milliseconds."""

import time

from hands.core.status import Stamp


def stamp_now() -> Stamp:
    return Stamp(time.time_ns() // 1_000_000)
