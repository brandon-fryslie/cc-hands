# fritter

fritter runs an interactive terminal program and gives it a second way in. The program
behaves exactly as it would on your own terminal — same interface, same width, same
signals, same exit code — and alongside it fritter listens on a unix socket. Anything
asked for on that socket is typed into the program's input, as though someone at the
keyboard had typed it.

```
fritter [--socket-dir DIR] -- COMMAND [ARGS...]
```

hands is what it was built for, and hands' client for it - `hands.sessions.typing` - is
written and tested against it; hands' `send_draft`, `send_command` and `interrupt_session`
are what dial it. Nothing in fritter knows any of that; Claude Code is simply the first
program it wraps.

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

The socket lives in a directory of its own, made 0700, and is itself 0600. Both are
needed. The default parent is whatever `TMPDIR` names, which is private for a login shell
but is the world-writable `/tmp` under launchd and cron — and a socket any local user can
dial is a socket any local user can type `!rm -rf ~` into. The directory goes when fritter
does.

macOS caps a unix socket path near 104 bytes. fritter checks the length where it picks
the path and says so, rather than letting `bind` fail with an error that names nothing.

## The protocol

One JSON object per connection, newline-terminated, answered with one JSON object.

```
{"pid":4242,"kind":"text","text":"fix the auth middleware"}
{"pid":4242,"kind":"key","key":"escape"}
```

`pid` is the process the caller means to type into, and a request naming any process but
the one fritter wrapped is refused. The address alone cannot say which session it
reaches: it is inherited, so a second session started from inside a wrapped one finds its
parent's address in its own environment. So wrap the program itself — `fritter -- claude`
— and not a launcher that runs it as a child. hands' `claude` shim, written by
`hands install-fritter`, does exactly that for every interactive session.

A `text` request is typed the way someone at the keyboard types a message: the text, then
Return. It goes into the child as one write. fritter does not decide what a leading `/` or
`@` means to the program underneath; that is the caller's, and in hands it is settled
before anything reaches here.

`text` is characters and newlines, and a control byte in it is refused by name: an `ESC`
would end the bracketing early and a `0x03` is a Ctrl-C. Send a `key` request for a
keystroke. The keys are `escape`, `enter`, `ctrl_c`, `ctrl_u`, `up`, `down`, `tab` and
`shift_tab`, and each is typed exactly as if the user had pressed it.

The answer is `{"ok":true}` once the write is done, or `{"ok":false,"reason":"..."}` when
the request was refused or the write failed. Writes into the child are made one at a time,
the user's keys and requests alike, so neither lands inside the other.

A caller has one second and 64 KB to get its request in.

## Multi-line text

fritter watches the child's output for the sequence that turns bracketed paste on, and
wraps injected text to match. A newline inside the text is then a newline in the message
rather than a Return that submits half of it. Text holding a newline is refused when the
program has not asked for bracketing, because it would arrive as several separate prompts.

## What run promises, and what it does not

Termination signals are forwarded to the child, so the ordinary exit path runs and the
socket is removed and the terminal restored. Without that, a killed fritter would die with
its cleanup unrun, leaving a stale socket for the next caller to dial into nothing.

Closing the terminal ends the session. The terminal's hangup reaches fritter as SIGHUP and
is forwarded like any other termination signal, and the child's output is read from the
pty for as long as the child writes, whether or not the terminal is still there to show
it. A child whose output nobody reads blocks on its way out — Claude Code waits in
`tcsetattr` for its output to drain — and would be left running, with fritter waiting on
it, after the window they ran in had gone.

The exit code is the child's. A child killed by a signal is reported as 128 plus that
signal, the way a shell reports it — SIGTERM is 143 — because a signalled child has no
exit code of its own, and passing on Go's -1 would exit 255 and leave a caller unable to
tell that from a program that really did exit 255.

`run` waits for the child's last output before returning. `cmd.Wait` comes back when the
child is reaped, and there is no ordering at all between that and the copy of what it
printed finishing. Returning there hands the caller a finished session whose last words
are still going out — and the caller's next move is to exit. Against a stdout that takes
its time the gap is plainly visible; against a fast one it is a race you win almost every
time, which is the worse kind. The wait stops after two seconds, because something the
child left running can hold the pty open and an unbounded wait would keep fritter alive
after its session ended. Reaching that bound means output really was lost, and fritter
says so.

The goroutine reading your stdin outlives `run`. `main` hands it `os.Stdin`, and a read
already blocked there cannot be interrupted — `SetReadDeadline` answers *"file type does
not support deadline"* for a terminal — so joining on it would hang the exit rather than
hurry it. (Measured, because an earlier version of this file had it backwards: a pty slave
*does* take a deadline and *does* unblock a read in flight. That is the shape the tests run
in, not the shape a session runs in, and a contract that holds only under test is not one.)
The caller must not close stdin before its process ends, which for fritter is the next
statement in `main`. A guarantee that sometimes deadlocks is worse than one not made.

## Measured against Claude Code

On 2.1.283:

- A bracketed paste and the Return behind it, read together, are sent: a short text and a
  3 KB one alike. A Return read a moment after the paste, apart from it, sent nothing, which
  is why the two go as one write.
- Read together that way, text ending on an `@` or `#` token is sent as it is: no
  completion list opens to take the Return. Text ending on a backslash is not sent. The
  Return becomes a newline, and the next text typed is submitted joined onto it.
- Text behind a leading space starting with `/`, `@` or `!` is sent as plain text, and the
  transcript records it with the space.
- The binary asks for bracketed paste.
- Killing the tmux session a wrapped session runs in, or sending fritter SIGHUP or
  SIGTERM, ends fritter, claude and the processes claude started within a second, and
  removes the socket directory.

On 2.1.278:

- Text wrapped in bracketed paste arrives as one multi-line message.
- A two-line prompt sent with the display asleep arrived as one message and was answered.
- A process the child spawns sees `FRITTER_SOCKET`.
- A permission dialog and the workspace-trust dialog each swallow a paste, and take the
  Return as their answer. A session sitting at a dialog is not one to type text into.

## Building and testing

```
cd fritter
go test -race ./...
go build -o fritter .
```

The tests wrap a shell, or the test binary itself reading its terminal raw and recording
each read, rather than Claude Code, so they need no session and no network.
They use short temporary directories on purpose: `t.TempDir()` names the directory after
the test, which pushes the socket path past the 104-byte limit.
