# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches AppKit.
"""`hands indicator`: the daemon's verdict as a menu-bar status item, in a process of its own.

It reads the heartbeat file and nothing else, so a daemon that hangs is shown as stuck, and one that dies is announced
by this process on its way out: it lives as long as the process that started it, `hands run` in a terminal, and then
until the heartbeat stops saying up; or until the run after a restart ends it for one of its own.
"""

import asyncio
import os
import signal
import threading
from datetime import UTC, datetime

import AppKit
from Foundation import NSRunLoop, NSRunLoopCommonModes, NSTimer
from loguru import logger
from PyObjCTools import AppHelper, MachSignals

from hands.daemon import indicator
from hands.sessions import heartbeat
from hands.daemon.notify import post_notification
from hands.sessions.home import Home

# How often the heartbeat is looked at: a turn opening, or a daemon dying, is shown within this of the file saying so.
LOOK_SECONDS = 0.2
# How long the notice posted on the way out may take before the indicator exits without it.
LAST_POST_SECONDS = 5.0
# An event with nothing in it, posted to wake the event loop once it is told to stop.
WAKE = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
    AppKit.NSEventTypeApplicationDefined, (0, 0), 0, 0, 0, None, 0, 0, 0
)


def show(home: Home, run: int) -> int:
    """Run the status item until `run`, the process that started this one, is gone and the heartbeat has said so; then
    0, the indicator's exit code. What made it stop looking at the heartbeat first is raised."""
    app = AppKit.NSApplication.sharedApplication()
    # A menu-bar item only: no Dock icon, no menu bar of its own, never the active app.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
    # No Quit item: the indicator goes when the run does, and the surface that says hands is down is not one click from gone.
    menu = AppKit.NSMenu.alloc().init()
    verdict_line = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
    verdict_line.setEnabled_(False)
    item.setMenu_(menu)
    before: indicator.Shown | None = None
    # How the indicator ends, set by the look that ends it: 0, the run gone and said so, or what the look raised.
    ending: int | Exception | None = None

    def end(how: int | Exception) -> None:
        nonlocal ending
        ending = how
        timer.invalidate()
        # An open menu tracks events in a loop of its own, which a posted event would wait behind until it closed.
        menu.cancelTracking()
        app.stop_(None)
        # stop_ takes effect once the event being handled is done, and a timer firing is no event: one posted wakes
        # the loop to end, so the event loop returns here and the command's event is written as it ends.
        app.postEvent_atStart_(WAKE, True)

    def look(_timer: object) -> None:
        nonlocal before
        try:
            now = datetime.now(UTC)
            verdict = heartbeat.look(home.status, now)
            seen = indicator.show(before, verdict, now)
            before = seen
            item.button().setTitle_(seen.title)
            item.button().setToolTip_(seen.text)
            verdict_line.setTitle_(seen.text)
            # Once `run` is gone, this process has been handed to another parent, and never back.
            if indicator.finished(verdict, orphaned=os.getppid() != run, run=run):
                # Posted before ending, not beside it: the notice that the run went is the last thing this process
                # does, bounded so that an osascript that never returns cannot keep the process up in its place.
                for notice in indicator.last_words(seen):
                    asyncio.run(post_last(notice))
                end(0)
                return
            for notice in seen.notices:
                post(notice)
        except Exception as error:
            # [LAW:no-silent-failure] AppKit would log a failed timer and carry on showing a stale light; ending
            # instead raises the failure into the terminal the run prints to, and takes the stale light away.
            end(error)

    # A restart's run ends this indicator for its own with SIGTERM: delivered on the run loop, it ends through `end` as
    # the run going does, so the command returns and its event is written.
    MachSignals.signal(signal.SIGTERM, lambda _signum: end(0))
    timer = NSTimer.timerWithTimeInterval_repeats_block_(LOOK_SECONDS, True, look)
    # The common modes include the one the run loop is in while the menu is open, so an open menu keeps up too.
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
    # The first look as the loop begins, rather than a period after: one that ends the indicator stops a running loop.
    AppHelper.callAfter(look, None)
    AppHelper.runEventLoop(installInterrupt=True)
    match ending:
        case int():
            return ending
        case Exception():
            raise ending
        case None:
            # [LAW:no-silent-failure] PyObjC prints a KeyboardInterrupt or SystemExit and returns from its loop.
            raise RuntimeError("the indicator's event loop ended before the run did")


def post(notice: str) -> None:
    """Post off the main thread: osascript takes a moment, and the status item must keep up meanwhile."""
    # post_notification logs its own failure; nothing here waits on it.
    threading.Thread(target=lambda: asyncio.run(post_notification(notice)), name="notice", daemon=True).start()


async def post_last(notice: str) -> None:
    """Post a notice on the way out, or say that it could not be posted in time; the indicator exits either way."""
    try:
        await asyncio.wait_for(post_notification(notice), LAST_POST_SECONDS)
    except TimeoutError:
        logger.error(f"osascript did not post {notice!r} within {LAST_POST_SECONDS:.0f}s; the indicator exits without it")
