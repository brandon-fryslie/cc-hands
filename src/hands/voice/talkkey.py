# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches Quartz.
"""The talk key's edge: Right Shift, and every other key, as macOS reports them from whichever app has focus.

A listen-only event tap: it sees keys and never holds one back, so Shift keeps working as Shift in every app. macOS
shows it keys only with the Input Monitoring grant, which belongs to the app hands runs in, hands.app or a terminal's; without the
grant no tap is made, so the grant is checked at the door (`granted`) and a tap that still cannot be made fails loudly.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

import Quartz
from loguru import logger

from hands.voice.hold import Instant, KeyEvent, Pressed, Released, Typed

# kVK_RightShift. Its presses and releases arrive as flag changes, not key downs, as every modifier's do.
RIGHT_SHIFT = 60
# NX_DEVICERSHIFTKEYMASK: set in an event's flags while Right Shift itself is down, whatever Left Shift is doing.
RIGHT_SHIFT_DOWN = 0x4
# Declared, so the strict code that uses it sees an int.
FLAGS_CHANGED: int = Quartz.kCGEventFlagsChanged
# Set in Right Shift's flags when it goes down inside a chord (Cmd+Shift+4 begun with Cmd): Left Shift's own bit
# (NX_DEVICELSHIFTKEYMASK), Control, Option, Command, and fn. Right Shift alone is never any of them.
OTHER_MODIFIERS: int = (
    0x2
    | Quartz.kCGEventFlagMaskControl
    | Quartz.kCGEventFlagMaskAlternate
    | Quartz.kCGEventFlagMaskCommand
    | Quartz.kCGEventFlagMaskSecondaryFn
)
# Everything else a hand can do while Right Shift is held, each of which makes it Shift: a key, a modifier, a click
# (shift-click extends a selection), a scroll (shift-scroll scrolls sideways).
WATCHED_KINDS: tuple[int, ...] = (
    Quartz.kCGEventKeyDown,
    Quartz.kCGEventFlagsChanged,
    Quartz.kCGEventLeftMouseDown,
    Quartz.kCGEventRightMouseDown,
    Quartz.kCGEventOtherMouseDown,
    Quartz.kCGEventScrollWheel,
)
# What macOS sends in place of an event when it has switched the tap off: after a callback it judged too slow, or at
# secure input. A key released meanwhile was never seen.
SWITCHED_OFF = (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput)


def granted() -> bool:
    """Whether macOS lets this process see keys typed in other apps."""
    return bool(Quartz.CGPreflightListenEventAccess())


def ask() -> None:
    """Put the app hands runs in on Input Monitoring's list in System Settings, and prompt for the grant once."""
    Quartz.CGRequestListenEventAccess()


def event_of(kind: int, keycode: int, flags: int, at: Instant) -> KeyEvent:
    """What one tapped event is to the hold: Right Shift going down alone or coming up, or anything else."""
    if kind == FLAGS_CHANGED and keycode == RIGHT_SHIFT:
        if not flags & RIGHT_SHIFT_DOWN:
            return Released()
        return Typed() if flags & OTHER_MODIFIERS else Pressed(at)
    return Typed()


def tap(heard: Callable[[KeyEvent], None]) -> Callable[[], None]:
    """Watch every key on a thread of its own, calling `heard` there with each; returns what stops the watch."""

    def callback(_proxy, kind, event, _refcon):
        if kind in SWITCHED_OFF:
            # [LAW:no-silent-failure] a release may have gone by unseen, so a turn in progress is dropped, not left
            # open, and the tap is switched back on. Nothing in hands switches the tap off, so this is always macOS.
            logger.warning("macOS switched the talk key's tap off; a turn in progress is dropped, and the tap is back on")
            Quartz.CGEventTapEnable(port, True)
            heard(Typed())
        else:
            keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            heard(event_of(kind, keycode, Quartz.CGEventGetFlags(event), time.monotonic()))
        return event

    watched = 0
    for kind in WATCHED_KINDS:
        watched |= Quartz.CGEventMaskBit(kind)
    port = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly, watched, callback, None
    )
    if port is None:
        raise RuntimeError("macOS refused to let hands watch the keyboard: grant Input Monitoring to the app hands runs in")
    source = Quartz.CFMachPortCreateRunLoopSource(None, port, 0)
    running: Future[object] = Future()

    def watch() -> None:
        run_loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(run_loop, source, Quartz.kCFRunLoopCommonModes)
        running.set_result(run_loop)
        Quartz.CFRunLoopRun()
        # [LAW:single-enforcer] the thread that runs the tap is the one that ends it, and only once its run loop has
        # stopped, so no callback can run after the watch is over.
        Quartz.CFMachPortInvalidate(port)

    watching = threading.Thread(target=watch, name="the talk key tap", daemon=True)
    watching.start()
    run_loop = running.result()

    def stop() -> None:
        # [LAW:no-ambient-temporal-coupling] the stop is handed to the tap's own run loop, which performs it whenever
        # it runs, even if that is only after this call: there is no flag to race and no window to miss.
        Quartz.CFRunLoopPerformBlock(run_loop, Quartz.kCFRunLoopCommonModes, lambda: Quartz.CFRunLoopStop(run_loop))
        Quartz.CFRunLoopWakeUp(run_loop)
        watching.join()

    return stop
