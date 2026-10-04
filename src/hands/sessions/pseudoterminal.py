"""A Claude Code run on a pseudo-terminal hands holds: read as fast as it draws, the last of what it showed kept, and
ended by hands' end however that comes.

The brain, each side question, and the smoke test's working session all run so; what each runs, and in what environment,
is theirs to say.
"""

import asyncio
import fcntl
import os
import pty
import re
import struct
import subprocess
import termios
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from hands.core.session import ESCAPES
from hands.sessions.child import Child, reaped

# The terminal a Claude Code of hands' own draws on. Nobody looks at it; it is sized so a long line is not wrapped into many.
ROWS, COLS = 50, 200
# How much of what it last showed is kept, for the line that says it exited or never came up.
SHOWN_BYTES = 16 * 1024
SHOWN_LINES = 20
# How long a Claude Code told to stop has before it is killed.
STOP_SECONDS = 5.0

# What a terminal is told rather than shown, and the keys it is sent.
_CONTROL = re.compile(rf"{ESCAPES.pattern}|[\x00-\x09\x0b-\x1f\x7f]")


class _Terminal:
    """The pseudo-terminal's master side, held by hands: read as fast as it is written, and the last of it kept."""

    def __init__(self, master: int) -> None:
        self._master = master
        self._shown = b""
        self._lock = threading.Lock()
        loop = asyncio.get_running_loop()
        self.closed: asyncio.Future[None] = loop.create_future()

        def read() -> None:
            # A thread of its own: nothing is drawn when nothing reads, and a terminal's reads block.
            while True:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    data = b""
                if not data:
                    break
                with self._lock:
                    self._shown = (self._shown + data)[-SHOWN_BYTES:]
            os.close(master)
            loop.call_soon_threadsafe(lambda: None if self.closed.done() else self.closed.set_result(None))

        threading.Thread(target=read, name="a Claude Code's terminal", daemon=True).start()

    def last(self) -> str:
        """The last lines it showed, as text."""
        with self._lock:
            shown = self._shown.decode(errors="replace")
        lines = [line.rstrip() for line in _CONTROL.sub("", shown.replace("\r", "\n")).split("\n") if line.strip()]
        return "\n".join(lines[-SHOWN_LINES:])


class ClaudeCode:
    """A Claude Code running on a terminal hands holds, until it ends or is stopped."""

    def __init__(self, child: Child[int], terminal: _Terminal) -> None:
        self._child = child
        self._terminal = terminal
        # Its exit code, once it has ended and what it showed last is read.
        self.exit = asyncio.ensure_future(self._run_out())

    @property
    def pid(self) -> int:
        return self._child.process.pid

    def shown(self) -> str:
        """The last lines it showed on its terminal."""
        return self._terminal.last()

    async def stop(self) -> None:
        await self._child.stopped(STOP_SECONDS)
        await asyncio.shield(self.exit)

    async def _run_out(self) -> int:
        code = await asyncio.shield(self._child.ended)
        try:
            # What it showed last, read to its end; a terminal some child of it still holds is not waited on for long.
            await asyncio.wait_for(asyncio.shield(self._terminal.closed), 1.0)
        except TimeoutError:
            pass
        return code


async def on_terminal(argv: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> ClaudeCode:
    """Run `argv` in `cwd` and `environment` on a pseudo-terminal of hands' own, as a terminal emulator runs a program."""
    master, slave = pty.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        # Not asyncio's subprocesses, for the reason `hands.sessions.child.Child` gives.
        process = subprocess.Popen(
            _holding_terminal(os.ttyname(slave), argv),
            cwd=cwd,
            env={**environment, "TERM": "xterm-256color"},
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    child = reaped("a Claude Code", process, process.wait)
    try:
        await _held(master, child)
    except BaseException:
        os.close(master)
        await child.killed("its start failed or was cut short")
        raise
    return ClaudeCode(child, _Terminal(master))


async def _held(terminal: int, child: Child[int]) -> None:
    """Until `child` holds `terminal` as its session's, or has ended without taking it.

    [LAW:no-ambient-temporal-coupling] what is spawned is ended by hands' end only once it holds its terminal: a hands
    that died before then would hang up nothing, and leave the child opening a terminal with no other end, for good."""
    while not child.ended.done() and os.tcgetpgrp(terminal) != child.process.pid:
        await asyncio.sleep(0.002)


def _holding_terminal(terminal: str, argv: Sequence[str]) -> list[str]:
    """argv, run so that its terminal is its session's controlling terminal: hands' end, however it comes, hangs the
    terminal up and ends what runs on it, as closing a window does. Without it, a hands that dies without stopping it
    leaves it running for good.

    A session leader with no controlling terminal takes the first terminal it opens, so the shell opens it and execs
    argv in its place, keeping its pid. [LAW:no-ambient-temporal-coupling] no Python runs between fork and exec, where
    a lock another of hands' threads held at the fork would hang the child, and hands with it."""
    return ["/bin/sh", "-c", ': <>"$0"; exec "$@"', terminal, *argv]
