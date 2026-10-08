"""`hands grant`: the Input Monitoring grant the talk key needs, given by the person in System Settings while this waits.

macOS gives the grant to an app, never to a process: to the app responsible for the process that asks, the terminal app
hands runs in or hands.app. Over ssh the responsible process is sshd's, which is no app, so there is nothing the person
could turn on, and this says so at once rather than wait.

Measured on macOS 26.6 (hands-fresh): once the person turns the app on and answers macOS's offer to quit it with Later,
a process the app starts after that has the grant, while one already running when it was given keeps the answer it had.
So the wait asks a new process each time, and the app is never quit, which would end this command with it.
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hands.sessions import audit, terminals, wide
from hands.voice import talkkey

# System Settings > Privacy & Security > Input Monitoring.
PANE = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
WAIT_SECONDS = 300
POLL_SECONDS = 1.0


@dataclass(frozen=True)
class App:
    """The app macOS gives the grant to, by the name System Settings lists it under."""

    name: str


@dataclass(frozen=True)
class NoApp:
    """A process responsible for hands that no app holds, as sshd's is over ssh: no grant can be given to it."""

    executable: str


type Grantee = App | NoApp


def grantee(executable: str) -> Grantee:
    """[LAW:parse-dont-validate] the app macOS gives the grant to, from the executable of the process responsible for
    hands: the innermost .app bundle around it, or none."""
    bundles = [part for part in Path(executable).parts if part.endswith(".app")]
    return App(bundles[-1].removesuffix(".app")) if bundles else NoApp(executable)


@dataclass(frozen=True)
class Mac:
    """What `hands grant` asks of macOS, so a test stands in for it: the executable of the process responsible for
    hands, whether a new process has the grant, macOS's own request (which puts the app on Input Monitoring's list), the
    pane opened, the wait between asks, and the clock the wait is bounded by."""

    responsible: Callable[[], str]
    granted: Callable[[], bool]
    ask: Callable[[], None]
    show: Callable[[], None]
    sleep: Callable[[float], None]
    now: Callable[[], float]


def on_this_mac() -> Mac:
    return Mac(responsible=lambda: terminals.responsible(os.getpid()), granted=granted_anew, ask=talkkey.ask, show=show_pane, sleep=time.sleep, now=time.monotonic)


def show_pane() -> None:
    subprocess.run(["open", PANE], check=True)


def granted_anew() -> bool:
    """Whether a process started now has the grant: this one's answer is the one it had when it first asked."""
    # [LAW:no-silent-failure] the answer is what the new process prints, so one that crashed is an error, never a no.
    answer = subprocess.run([sys.executable, "-c", "from hands.voice.talkkey import granted; print(granted())"], stdout=subprocess.PIPE, text=True, check=True).stdout
    match answer:
        case "True\n":
            return True
        case "False\n":
            return False
        case _:
            raise ValueError(f"a new process asked whether it has the Input Monitoring grant answered {answer!r}")


def give(mac: Mac, record: audit.Record) -> int:
    """The grant, already there or given while this waits: exits 0 once it is; 1 when it is not given within the wait,
    which running this again waits for again; 2 when no app holds hands, so it cannot be given here at all."""
    # [LAW:nothing-unseen] a grant is a unit of work: the app it goes to, whether it was there before, how many times a
    # new process was asked while it waited, and whether it was there after.
    with wide.unit("talkkey.grant", record):
        to = grantee(mac.responsible())
        before = mac.granted()
        wide.annotate(grantee=to, before=before)
        match before, to:
            case True, App(name=name):
                print(f"{name}, the app hands runs in, has the Input Monitoring grant, so hands hears the talk key (Right Shift)")
                return 0
            case True, NoApp():
                print("hands has the Input Monitoring grant, so it hears the talk key (Right Shift)")
                return 0
            case False, NoApp(executable=executable):
                return not_granted(f"hands runs here under {executable}, which is no app, and macOS gives the Input Monitoring grant only to an app, as it cannot over ssh: run this in a terminal on the Mac itself", 2)
            case False, App(name=name):
                return waited(name, mac)


def waited(name: str, mac: Mac) -> int:
    """Asks macOS for the grant to `name`, opens its pane, says what to turn on, and waits for a new process to have it."""
    mac.ask()
    mac.show()
    print(
        f"hands hears the talk key (Right Shift) in other apps only with macOS's Input Monitoring grant, which goes to {name}, the app hands runs in. "
        f"System Settings is open at Privacy & Security > Input Monitoring: turn {name} on. When macOS offers to quit {name}, choose Later: hands has the grant without it, "
        f"and quitting {name} ends this. Waiting up to {WAIT_SECONDS // 60} minutes.",
        flush=True,
    )
    polls = 0
    deadline = mac.now() + WAIT_SECONDS
    while mac.now() < deadline:
        mac.sleep(POLL_SECONDS)
        polls += 1
        if mac.granted():
            wide.annotate(polls=polls, after=True)
            print(f"{name} has the Input Monitoring grant, so hands hears the talk key (Right Shift)")
            return 0
    wide.annotate(polls=polls, after=False)
    return not_granted(f"{name} was not given the Input Monitoring grant within {WAIT_SECONDS // 60} minutes; running this again waits again", 1)


def not_granted(said: str, status: int) -> int:
    wide.fail(said)
    print(f"hands grant: {said}", file=sys.stderr)
    return status
