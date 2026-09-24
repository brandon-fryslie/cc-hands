# pyright: basic, reportAttributeAccessIssue=false
# PyObjC ships no type stubs and loads its names lazily, so a static check cannot see them; this file is the only one
# that touches AppKit.
"""`hands indicator`: the daemon's verdict as a menu-bar status item, in a process of its own.

It reads the heartbeat file and nothing else, so whatever becomes of the daemon — dead, hung, never started — this
process is still here to show it, whichever terminal the user runs or none.
"""

import asyncio
import os
import threading
from datetime import UTC, datetime

import AppKit
from Foundation import NSTimer
from loguru import logger
from PyObjCTools import AppHelper

from hands.daemon import indicator, status
from hands.daemon.notify import post_notification
from hands.sessions.home import Home

# How often the heartbeat is looked at: a daemon that dies is shown within this of the verdict changing.
LOOK_SECONDS = 1.0


def show(home: Home) -> None:
    """Run the status item until the process is told to stop."""
    app = AppKit.NSApplication.sharedApplication()
    # A menu-bar item only: no Dock icon, no menu bar of its own, never the active app.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSVariableStatusItemLength)
    # No Quit item: launchd keeps the indicator up, and the surface that says hands is down is not one click from gone.
    menu = AppKit.NSMenu.alloc().init()
    verdict_line = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
    verdict_line.setEnabled_(False)
    item.setMenu_(menu)
    before: indicator.Shown | None = None

    def look(_timer: object) -> None:
        nonlocal before
        try:
            now = datetime.now(UTC)
            seen = indicator.show(before, status.look(home.status, now), now)
            before = seen
            item.button().setTitle_(seen.title)
            item.button().setToolTip_(seen.text)
            verdict_line.setTitle_(seen.text)
            for notice in seen.notices:
                post(notice)
        except Exception:
            # [LAW:no-silent-failure] AppKit would log a failed timer and carry on showing a stale light; exiting
            # instead is seen in the log and ends in launchd starting a fresh indicator.
            logger.exception("the indicator failed to look at the heartbeat")
            os._exit(1)

    look(None)
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(LOOK_SECONDS, True, look)
    AppHelper.runEventLoop(installInterrupt=True)


def post(notice: str) -> None:
    """Post off the main thread: osascript takes a moment, and the status item must keep up meanwhile."""
    # post_notification logs its own failure; nothing here waits on it.
    threading.Thread(target=lambda: asyncio.run(post_notification(notice)), name="notice", daemon=True).start()
