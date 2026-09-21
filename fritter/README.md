# fritter

fritter runs an interactive terminal program and gives it a second way in. The program
behaves exactly as it would on your own terminal — same interface, same width, same
signals, same exit code — and alongside it fritter listens on a unix socket. Anything
asked for on that socket is typed into the program's input, as though someone at the
keyboard had typed it.

```
fritter [--socket-dir DIR] -- COMMAND [ARGS...]
```

hands uses it to drive Claude Code sessions. Nothing in fritter knows that; Claude Code
is simply its first caller.

## Why a pseudo-terminal and not a pipe

Claude Code asks whether its input is a terminal, and takes a different, non-interactive
path when it is not. A wrapper built on pipes would drive a different program than the
one you are trying to reach. So fritter allocates a pty, runs the command on the slave
side, and holds the master.

That choice is also what makes it work with the screen off. There is no window to focus,
no keystroke to synthesise, and no Accessibility or Input Monitoring grant to ask for —
just bytes written to a file descriptor. It works over SSH and with the lid shut.

## Finding the socket

fritter chooses the socket's path and publishes it to the child in `FRITTER_SOCKET`.
Anything the child spawns inherits that variable, which is the whole of fritter's
coupling to whatever drives it.

That is how hands finds a session. Claude Code runs the hands hook as a child of the
session process, the hook reads `FRITTER_SOCKET` out of its own environment, and the
address goes into the session's membership file. Neither side has to agree on a filename
or derive one from a pid.

macOS caps a unix socket path near 104 bytes. fritter checks the length where it picks
the path and says so, rather than letting `bind` fail with an error that names nothing.

## The protocol

One JSON object per connection, newline-terminated, answered with one JSON object.

```
{"kind":"text","text":"fix the auth middleware","submit":true}
{"kind":"key","key":"escape"}
```

The answer is `{"ok":true}` or `{"ok":false,"reason":"..."}`. A reason always says what
went wrong, because a write that was refused and a write that landed must never look
alike to the caller.

`text` is typed literally. fritter does not decide what a leading `/` or `@` means to the
program underneath — that belongs to the caller, and in hands it is already settled
before anything reaches here. `submit` presses Enter afterwards.

The keys are `escape`, `enter`, `ctrl_c`, `up`, `down`, `tab` and `shift_tab`. Which
bytes each one is, is terminal knowledge, so that table lives here rather than in the
caller.

## Multi-line text

fritter watches the child's output for the sequence that turns bracketed paste on, and
wraps injected text to match. A newline inside the text is then a newline in the message
rather than the Enter that submits a half-written one.

This is read from the child rather than assumed, so a program that does not want
bracketing gets its text bare.

## Who owns the input line

If the person at the keyboard has typed something they have not submitted, fritter
refuses to write and says so. Their half-written line is theirs until they press Enter or
Ctrl-C.

This is the only fact about the input box fritter owns. Whether a session should be
written to at all — whether it is working, or sitting at a permission dialog — is the
caller's to decide, and hands decides it from state fritter cannot see. fritter's rule
covers the one thing the caller cannot know, because those keystrokes never reach it.

Escape also clears the box and is deliberately not treated as a clear: it is equally the
first byte of every arrow key, so treating it that way would free the line every time the
user pressed Up. The cost is that someone who presses Escape and walks away keeps the line
held until they press Enter or Ctrl-C. A refused write is loud and recoverable; a write
into a half-typed line is a garbled prompt nobody can attribute.

## What run promises, and what it does not

Termination signals are forwarded to the child, so the ordinary exit path runs and the
socket is removed and the terminal restored. Without that, a killed fritter would die with
its cleanup unrun, leaving a stale socket for the next caller to dial into nothing.

The exit code is the child's. A child killed by a signal is reported as 128 plus that
signal, the way a shell reports it — SIGTERM is 143 — because a signalled child has no
exit code of its own, and passing on Go's -1 would exit 255 and leave a caller unable to
tell that from a program that really did exit 255.

The goroutine reading your stdin outlives `run`. A read already blocked on a terminal
cannot be interrupted portably — `SetReadDeadline` answers *"file type does not support
deadline"* for a pty slave on macOS, and where it returns nil it does not reliably unblock
a read in flight — so joining on it would hang the exit rather than hurry it. The caller
must not close stdin before its process ends, which for fritter is the next statement in
`main`. A guarantee that sometimes deadlocks is worse than one not made.

## Measured against Claude Code 2.1.278

- The child turns bracketed paste on, and text wrapped in it arrives as one multi-line
  message.
- Text followed immediately by Enter submits correctly. No delay is needed between them,
  so there is no timing bet in the send path.
- A process the child spawns sees `FRITTER_SOCKET`.
- The workspace-trust dialog swallows a paste, the same way `docs/architecture.md` records
  permission dialogs doing. A session sitting at a dialog is not one to type text into.

## Building and testing

```
cd fritter
go test -race ./...
go build -o fritter .
```

The tests wrap a shell rather than Claude Code, so they need no session and no network.
They use short temporary directories on purpose: `t.TempDir()` names the directory after
the test, which pushes the socket path past the 104-byte limit.
