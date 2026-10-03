"""A child process run to its end inside a time limit, and never left behind by a caller that stops waiting on it."""

import asyncio
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ran:
    """How a child exited, and everything it wrote."""

    returncode: int
    out: bytes
    err: bytes


async def run(*argv: str, timeout: float, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> Ran:
    """Run `argv` to its exit; raises TimeoutError when it runs past `timeout`, and OSError when it cannot start.

    [LAW:single-enforcer] the one place hands runs a child to completion: timed out or its caller cancelled, the
    process is killed and reaped before the error leaves, so the loop never reports a subprocess still running.
    Each caller says what a timeout means for it.

    Not asyncio's subprocesses, whose Python 3.12 transport cannot be cancelled while it starts: its exit is
    queued behind a task of its own that the loop's shutdown cancels alongside, and the shutdown then waits for
    ever on an exit nothing will deliver. A loop closing as git started never finished closing. So the child is
    started here, its output goes to files, and its exit is waited for on a thread — which is what asyncio's own
    child watcher does on macOS — that the kill below always releases.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(argv, cwd=cwd, env=None if env is None else dict(env), stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        try:
            returncode = await asyncio.to_thread(process.wait, timeout)
        except BaseException as error:
            process.kill()
            # Reaped here and not on the thread, which may never be scheduled again once the loop is closing; a
            # killed child exits at once, so this is no wait the loop would notice.
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError(f"{argv[0]} ran past {timeout:.1f}s") from None
            raise
        out.seek(0)
        err.seek(0)
        return Ran(returncode, out.read(), err.read())
