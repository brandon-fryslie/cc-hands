# Testing

hands is proven by tests that run in `make check` on any Mac. The tests talk to the
outside world only through designed interfaces, and every interface has a fake that a
test can put in any state. No acceptance criterion depends on a particular machine, a
fresh user account, a real microphone, or a person clicking through a dialog.

This document describes those interfaces, the fakes, and the rule for when a ticket is
done. It extends the split described in [architecture.md](architecture.md): `core` is
pure and is tested with plain values, and the edges perform effects. This document is
about the edges. Most of it is not built yet: the "Today" column below says where each
interface stands, and the epic `promptctl-testing-c3u` tracks the work.

## Why not a copy of production

Until now, the final proof for several tickets was a run on `hands-fresh`, a tart VM
that stood in for a new Mac: install hands, click through macOS's permission dialogs,
sign in, and speak a turn. `hands smoke` is the same idea in one command, and its own
docstring says it "is run by hand on an installed Mac". That kind of proof has three
problems.

- It mostly tests other people's code. A permission dialog is Apple's, a login screen
  is Claude Code's, and a license response is Polar's. hands cannot change any of
  them, and running them again for every ticket says almost nothing about the change
  under review.
- It cannot reach the states that matter. A permission the administrator restricted, a
  `tccutil` reset that fails, a session whose process ended between two reads, or a
  fritter that never answers are hard or impossible to produce on a real machine, and
  they are exactly where hands' own logic can be wrong.
- It needs a spare Mac and a person, so it runs rarely, and a ticket waits on it.

What hands owns is its response to every answer the outside world can give. That
response is testable when the outside world is an interface with a fake behind it.

## The rule for done

A ticket is done when the behavior it describes is asserted by tests in `make check`,
at the interfaces below. "On hands-fresh", "on this Mac", and "heard live" are not
acceptance criteria. A ticket that needs a new kind of outside behavior adds it to the
interface and its fake first, then tests against the fake.

## Interfaces

Each party outside hands has one interface. Exactly one adapter makes the real calls,
and only the composition roots, `hands.daemon.run` and `app/main.swift`, connect real
adapters. Every other module receives the interface as a parameter
`[LAW:effects-at-boundaries]`. A boundary test enforces this the way
`tests/test_core_boundary.py` enforces that `core` is pure: the modules that reach the
real party (for example `subprocess`, `Quartz`, `pyaudio`, `AVFoundation`) may be
imported only by their adapter. The test arrives with the first interface, and each
later interface adds its modules to it.

| Interface | What it covers | Today |
|---|---|---|
| `Clock` | Wall time, monotonic time, sleeping, scheduling, ids, and randomness | `Sessions(clock=, stamp=)` and a few other `clock=` parameters. More than 40 call sites read the time directly, and about 200 test sites wait on real time. |
| `Processes` | The process table, a process's start time, cwd and arguments, signals, and child processes | `still_running` is pure, but `process_table()` and `child.run` are called directly. Nine tests fail in a sandbox because they scan the real process table. |
| `Terminals` | tmux servers, panes and typing, and the fritter control socket | `Sessions(typist=, keyboards=)`. Below that, every pane, capture, and typing behavior needs a real tmux. |
| `ClaudeCode` | Status files, transcripts, hooks, the `claude` CLI (`auth`, `plugin`, first run), and `.claude.json` | Files are read through `Home` and `Membership`. The CLI is replaced by stand-in scripts on `PATH`. |
| `Audio` | Microphone frames in, speaker frames out, and the default-device changes | The `PortAudio` and `Stream` protocols. `build_voice` does not pass them, so the voice tests still start the real PortAudio. |
| `Keys` | The talk key and the Input Monitoring status and request | Tests patch `talkkey.tap`, `talkkey.granted`, and `talkkey.ask`. `grant.Mac` is the one declared seam. |
| `Speech` | LowTalker transcription in, text to speech out | `Whisper(url=)` against a local stand-in server. Tests patch `pipeline.PocketTTSService`. |
| `Upstream` | The Anthropic API behind the wire proxy | `serve_proxy(upstream=)`. The composition root hard-codes the real URL. |
| `Polar` | License key validation | The app's Info.plist URL, pointed at a local stand-in server. |
| `Privacy` (Swift) | A permission's status, request, and reset | `HANDS_APP_PERMISSIONS` replaces the whole `hands --permissions` process. The real side's decisions are never tested. |
| `Shell` and `Daemon` (Swift) | The login shell's environment, and starting, stopping, and the exit of `hands run` | `HANDS_APP_SHELL`, with a stand-in `hands` script on the stand-in shell's `PATH`. |
| `Installer` | Homebrew, uv, the release lookup, and the steps `install.sh` runs | Stand-in `brew`, `curl`, `uv`, `sudo`, and `hands` scripts on `PATH`. The stand-in `hands` and the real subcommands agree only by convention. |

Where an interface already exists, it keeps its name and shape, and the work is to make
it the only route to the real party.

## Fakes

A fake is an in-memory implementation of an interface. A test sets its state, drives
hands, and reads what hands asked of it. A fake never sleeps and never touches the
machine, so a test that uses only fakes gives the same result on every run and every
Mac.

A fake knows how the real party behaves only from recordings. A recording is something
captured once from the real party and kept as a fixture, the way
`tests/fixtures/status/*.json` holds status files copied from a live session. Examples
are Claude Code's first-run screens and its `plugin list --json` output, Polar's
responses for a live, a revoked, and an expired key, and the four states macOS reports
for a permission. Each recording names its source and version. When the real party
changes or turns out to behave differently, that is a bug report: the fix is a new
recording and a test that uses it, never a manual run.

The virtual clock is the fake for `Clock`. A test advances it explicitly, so a timeout,
a deadline warning, or a hold that becomes a turn happens when the test says so, not
after a real wait.

## Contract tests

A fake is only useful while it behaves like the real party. Each interface has one
contract suite, a set of tests that describe the interface's behavior. The suite runs
against the fake. Where the real party is a local program, it also runs against the real
adapter: `git`, `tmux` on a private socket, the fritter binary, and the real `hands`
subcommands that `install.sh` calls. If the fake and the real adapter disagree, the
contract suite fails.

Some real parties cannot run in a test: macOS permissions, the CGEventTap, PortAudio's
devices, and network services. Their adapters are kept free of decisions. Each is a
direct call to a documented API, short enough to check by reading it against that
documentation. Any decision about the answer, for example "a denied permission is reset
and then requested again", lives above the adapter and is tested against the fake.

## The app

hands.app is split into a logic library and a thin AppKit layer. The library holds every
decision the app makes:

- the license verdict and the grace period
- the setup flow: which step is shown, what Next does for each access state, when the
  window advances, and when the app stops
- the launcher's phases, from licensing to running to quitting

The library is tested with `swift-testing` against fakes of `Privacy`, `Polar`, `Shell`,
`Daemon`, and `Clock`. The AppKit layer only shows the library's state and passes clicks
back to it, so a click on Next is a call on the library, and the tests call it directly.
`tests/test_app.py` keeps the tests that need the compiled app as a process: signals, the
log file, and the environment the login shell passes to hands.

## The fake world

The fake world is one fixture that connects every fake and runs the real composition:
`hands.daemon.run` for the daemon, and the app's logic library for the app. A test
scripts the world and asserts what hands did. For example, a test can take these steps
and then assert what was spoken and what was typed:

1. The talk key goes down.
2. Recorded audio frames arrive from the microphone.
3. LowTalker returns the text "run the tests".
4. The fake upstream returns the brain's reply.
5. The reply is typed into a fritter session.

The fake world replaces `hands smoke` and `hands-fresh` as the end-to-end proof. It runs
in `make check`, so every change is checked against the whole path, not only against its
own unit.

## Order of work

1. `Privacy` and the app's logic library, then `Polar`, `Shell`, and `Daemon`. This
   closes the permission and license tickets and gives the app's install and update
   tickets acceptance criteria that tests can check.
2. `ClaudeCode` and `Installer` recordings, so the install checkpoint becomes tests.
3. `Clock`, which removes the waits on real time and the flakiness they cause.
4. `Processes` and `Terminals`, which make the suite pass in a sandbox.
5. `Audio`, `Keys`, `Speech`, and `Upstream`, and then the fake world.
