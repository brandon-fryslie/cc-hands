"""A child process run to its end inside a time limit, and never left behind by a caller that stops waiting on it."""

import asyncio
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from loguru import logger

from hands.threads import settles_off_loop


@dataclass(frozen=True)
class Child[T]:
    """A child started in a session of its own, and the outcome a daemon thread of its own settles once it has reaped it.

    [LAW:single-enforcer] the one place hands ends a child it started: stopped, killed, or its stopper cancelled, the
    child is reaped before the call returns or raises, so the loop never ends with a subprocess of hands' still running.

    Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is queued
    behind a task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for ever on an exit
    nothing will deliver. A thread blocks on the exit instead: no task for the shutdown to cancel, no executor for the
    exit to join, and the loop never blocks.
    """

    name: str
    process: subprocess.Popen[bytes]
    ended: asyncio.Future[T]

    async def stopped(self, grace: float) -> None:
        """Ask the child alone to end, and kill everything it started once it has not within `grace`."""
        self.process.terminate()
        try:
            await asyncio.wait((self.ended,), timeout=grace)
        except asyncio.CancelledError:
            await self.killed("its stopper stopped waiting")
            raise
        if not self.ended.done():
            await self.killed(f"it had not ended {grace:.1f}s after it was asked to")

    async def killed(self, why: str) -> None:
        """Kill the child's process group, and wait while its thread reaps it."""
        pid = self.process.pid
        try:
            os.killpg(pid, signal.SIGKILL)
            logger.info(f"killed {self.name} ({pid}) and everything it started, because {why}")
        except (ProcessLookupError, PermissionError):
            # Every process in the group had exited: gone, ProcessLookupError; exited and not yet reaped, macOS says
            # PermissionError. Nothing was killed, and the log says so.
            logger.info(f"{self.name} ({pid}) had already ended when {why}")
        # Reaped by its own thread, which the kill releases; the loop stays live while a child the kill cannot end at
        # once, one in uninterruptible I/O, finishes dying. A caller told twice to stop leaves without waiting, and
        # the thread reaps it all the same.
        await asyncio.wait((self.ended,))


def reaped[T](name: str, process: subprocess.Popen[bytes], outcome: Callable[[], T]) -> Child[T]:
    """`process`, with `outcome` run by a thread of its own, which must reap it, to settle `ended`."""
    return Child(name, process, settles_off_loop(outcome, name=f"{name} {process.pid}"))


@dataclass(frozen=True)
class Ran:
    """How a child exited, and everything it wrote."""

    returncode: int
    out: bytes
    err: bytes


async def run(*argv: str, timeout: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Ran:
    """Run `argv` to its exit; raises TimeoutError when it runs past `timeout`, and OSError when it cannot start.

    The one place hands runs a child to completion: timed out or its caller cancelled, the child and every process
    it started are killed, and the child reaped, before the error leaves. Each caller says what a timeout means for
    it. Its output goes to files, which the child's own thread reads once it has reaped it.
    """
    out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    try:
        # A session of its own, so its process group is everything it starts: a git's filters, a lit's helpers.
        process = subprocess.Popen(
            argv, cwd=cwd, env=None if env is None else dict(env), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True
        )
    except BaseException:
        out.close()
        err.close()
        raise
    # The files are the thread's from here, and it closes them: a caller that leaves early cannot pull them from
    # under the read.
    child = reaped(argv[0], process, lambda: _ended(process, out, err))
    started = time.monotonic()
    try:
        # Waited on, not awaited: what the child's thread raised is the child's outcome, told below as its own.
        await asyncio.wait((child.ended,), timeout=timeout)
    except asyncio.CancelledError:
        await child.killed("its caller stopped waiting")
        raise
    if not child.ended.done():
        await child.killed(f"it ran past {timeout:.1f}s")
        raise TimeoutError(f"{argv[0]} ran past {timeout:.1f}s")
    ran = child.ended.result()
    logger.debug(f"{argv[0]} ({process.pid}) exited {ran.returncode} in {time.monotonic() - started:.3f}s")
    return ran


def _ended(process: subprocess.Popen[bytes], out: IO[bytes], err: IO[bytes]) -> Ran:
    with out, err:
        returncode = process.wait()
        out.seek(0)
        err.seek(0)
        return Ran(returncode, out.read(), err.read())
