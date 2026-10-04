"""The keyboard edges: Right Shift held alone, from any app, moves the turn; `q` typed in hands' own terminal quits."""

import asyncio
import sys
import termios
import tty
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager

from hands.voice import talkkey
from hands.voice.trigger import Trigger
from hands.voice.hold import HOLD_SECONDS, TURN_LIMIT_SECONDS, Hold, Idle, KeyEvent, Move, Overlong, Pressed, Ripe, step

QUIT = "q"


async def drive_talk_key(on_move: Callable[[Move], Awaitable[None]], trigger: Callable[[], Trigger]) -> None:
    """Hand every move of the turn to `on_move`, as the talk key makes them under the trigger in use, until cancelled."""
    loop = asyncio.get_running_loop()
    heard: asyncio.Queue[KeyEvent] = asyncio.Queue()

    def arrived(event: KeyEvent) -> None:
        heard.put_nowait(event)
        # [LAW:no-ambient-temporal-coupling] every press asks to be told when it has been held for HOLD_SECONDS and
        # for TURN_LIMIT_SECONDS, counted from the press itself on the loop's own clock, the monotonic one; the hold is
        # the one owner of what that means, and ignores the ones a release has made stale.
        match event:
            case Pressed(at=at):
                loop.call_at(at + HOLD_SECONDS, heard.put_nowait, Ripe(at))
                loop.call_at(at + TURN_LIMIT_SECONDS, heard.put_nowait, Overlong(at))
            case _:
                pass

    def heard_on_the_tap(event: KeyEvent) -> None:
        loop.call_soon_threadsafe(arrived, event)

    stop = talkkey.tap(heard_on_the_tap)
    hold: Hold = Idle()
    try:
        while True:
            event = await heard.get()
            # [LAW:one-type-per-behavior] read at every event, so a switch by voice holds from the next key the user presses.
            match trigger():
                case "held key":
                    hold, moves = step(hold, event)
            for move in moves:
                await on_move(move)
    finally:
        stop()


@contextmanager
def raw_terminal(fd: int) -> Generator[None]:
    """Put the terminal in cbreak mode for the duration and always restore it."""
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def drive_quit(quit_event: asyncio.Event) -> None:
    """Read hands' own terminal until `q`, which stops the run."""
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    typed: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(fd, lambda: typed.put_nowait(sys.stdin.read(1)))
    try:
        with raw_terminal(fd):
            while await typed.get() != QUIT:
                pass
            quit_event.set()
    finally:
        loop.remove_reader(fd)
