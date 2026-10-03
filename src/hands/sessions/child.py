"""A child process run to its end inside a time limit, and never left behind by a caller that stops waiting on it."""

import asyncio
import contextlib
import os
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from loguru import logger

from hands.threads import settles_off_loop


@dataclass(frozen=True)
class Ran:
    """How a child exited, and everything it wrote."""

    returncode: int
    out: bytes
    err: bytes


async def run(*argv: str, timeout: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Ran:
    """Run `argv` to its exit; raises TimeoutError when it runs past `timeout`, and OSError when it cannot start.

    [LAW:single-enforcer] the one place hands runs a child to completion: timed out or its caller cancelled, the
    child and every process it started are killed, and the child reaped, before the error leaves, so the loop
    never reports a subprocess still running. Each caller says what a timeout means for it.

    Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is
    queued behind a task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for
    ever on an exit nothing will deliver. A loop closing as git started never finished closing. So the child is
    started here, its output goes to files, and a daemon thread of its own blocks on its exit and reads what it
    wrote — no task for the shutdown to cancel, no executor for the exit to join, and the loop never blocks.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        # A session of its own, so its process group is everything it starts: a git's filters, a lit's helpers.
        process = subprocess.Popen(
            argv, cwd=cwd, env=None if env is None else dict(env), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
        )
        ended = settles_off_loop(lambda: _ended(process, out, err), name=f"child {argv[0]} {process.pid}")
        try:
            return await asyncio.wait_for(asyncio.shield(ended), timeout)
        except BaseException as error:
            # The group outlives a leader that has already exited; one already gone has nothing left to kill.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            why = f"it ran past {timeout:.1f}s" if isinstance(error, TimeoutError) else "its caller stopped waiting"
            logger.info(f"killed {argv[0]} ({process.pid}) and everything it started, because {why}")
            # Reaped by its own thread, which the kill releases; the loop stays live while a child the kill cannot
            # end at once, one in uninterruptible I/O, finishes dying.
            await asyncio.shield(ended)
            if isinstance(error, TimeoutError):
                raise TimeoutError(f"{argv[0]} ran past {timeout:.1f}s") from None
            raise


def _ended(process: subprocess.Popen[bytes], out: IO[bytes], err: IO[bytes]) -> Ran:
    returncode = process.wait()
    out.seek(0)
    err.seek(0)
    return Ran(returncode, out.read(), err.read())
