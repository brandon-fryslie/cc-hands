"""The keyboard edge: turns terminal keystrokes into key positions.

A terminal cannot report key-up, so the spike uses a toggle: one press of the
space bar is key-down, the next is key-up. `q` ends the run. This is the only
place that touches the terminal.
"""

import asyncio
import sys
import termios
import tty
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager

from hands.voice.ptt import Key

SPACE = " "
QUIT = "q"


@contextmanager
def raw_terminal(fd: int) -> Generator[None]:
    """Put the terminal in cbreak mode for the duration and always restore it."""
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def drive_key(
    on_key: Callable[[Key], Awaitable[None]],
    quit_event: asyncio.Event,
) -> None:
    """Read keystrokes until `q`; each space bar press flips the key position."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    queue: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(fd, lambda: queue.put_nowait(sys.stdin.read(1)))
    position: Key = "up"
    try:
        with raw_terminal(fd):
            while True:
                ch = await queue.get()
                if ch == QUIT:
                    quit_event.set()
                    return
                if ch == SPACE:
                    position = "down" if position == "up" else "up"
                    await on_key(position)
    finally:
        loop.remove_reader(fd)
