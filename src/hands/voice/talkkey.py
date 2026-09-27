# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches Quartz.
"""The talk key's edge: Right Shift, and every other key, as macOS reports them from whichever app has focus.

A listen-only event tap: it sees keys and never holds one back, so Shift keeps working as Shift in every app. macOS
shows it keys only with the Input Monitoring grant, which belongs to the app hands runs in, the terminal; without the
grant no tap is made, so the grant is checked at the door (`granted`) and a tap that still cannot be made fails loudly.
"""

import threading
import time
from collections.abc import Callable

import Quartz
from loguru import logger

from hands.voice.hold import Instant, KeyEvent, Pressed, Released, Typed

# kVK_RightShift. Its presses and releases arrive as flag changes, not key downs, as every modifier's do.
RIGHT_SHIFT = 60
# NX_DEVICERSHIFTKEYMASK: set in an event's flags while Right Shift itself is down, whatever Left Shift is doing.
RIGHT_SHIFT_DOWN = 0x4
# Declared, so the strict code that uses them sees ints: the event kinds the tap watches, and the flag every Shift sets.
KEY_DOWN: int = Quartz.kCGEventKeyDown
FLAGS_CHANGED: int = Quartz.kCGEventFlagsChanged
SHIFT: int = Quartz.kCGEventFlagMaskShift
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
    """What one tapped event is to the hold: Right Shift going down or up, or any other key."""
    if kind == FLAGS_CHANGED and keycode == RIGHT_SHIFT:
        return Pressed(at) if flags & RIGHT_SHIFT_DOWN else Released()
    return Typed()


def tap(heard: Callable[[KeyEvent], None]) -> Callable[[], None]:
    """Watch every key on a thread of its own, calling `heard` there with each; returns what stops the watch."""
    port = None

    def callback(_proxy, kind, event, _refcon):
        if kind in SWITCHED_OFF:
            # [LAW:no-silent-failure] a release may have gone by unseen, so a turn in progress is dropped, not left
            # open, and the tap is switched back on.
            logger.warning("macOS switched the talk key's tap off; a turn in progress is dropped, and the tap is back on")
            Quartz.CGEventTapEnable(port, True)
            heard(Typed())
        else:
            keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            heard(event_of(kind, keycode, Quartz.CGEventGetFlags(event), time.monotonic()))
        return event

    watched = Quartz.CGEventMaskBit(KEY_DOWN) | Quartz.CGEventMaskBit(FLAGS_CHANGED)
    port = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly, watched, callback, None
    )
    if port is None:
        raise RuntimeError("macOS refused to let hands watch the keyboard: grant Input Monitoring to the app hands runs in")
    source = Quartz.CFMachPortCreateRunLoopSource(None, port, 0)
    running: list[object] = []
    started = threading.Event()

    def watch() -> None:
        running.append(Quartz.CFRunLoopGetCurrent())
        Quartz.CFRunLoopAddSource(running[0], source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(port, True)
        started.set()
        Quartz.CFRunLoopRun()

    threading.Thread(target=watch, name="the talk key tap", daemon=True).start()
    started.wait()

    def stop() -> None:
        Quartz.CGEventTapEnable(port, False)
        Quartz.CFRunLoopStop(running[0])

    return stop
