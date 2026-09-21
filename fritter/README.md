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
written and tested against it. The effect that will call that client is not built yet
(`hands-harness-5nb.l0u`), so nothing in hands dials this socket today. Nothing in fritter
knows any of that; Claude Code is simply the first program it wraps.

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
{"kind":"text","text":"fix the auth middleware","submit":true}
{"kind":"key","key":"escape"}
```

The answer is `{"ok":true}` or `{"ok":false,"reason":"..."}`. A reason always says what
went wrong, because a write that was refused and a write that landed must never look
alike to the caller. When the text lands but the Enter after it does not, the reason says
so in those words — retyping text that is already sitting in the box would double it.

A caller has one second and 64 KB to get its request in. Past the size it is refused with
a reason that says so; past the time the connection is dropped, because a client that
connects and never finishes its line would otherwise hold a goroutine for the rest of the
session. Both are generous for a local client sending one small object.

Each phase of an exchange is bounded on its own, and that is deliberate. Reading the
request gets **one** second, a write into the session gives up after **one** so its reason
has time to be written, and the reply gets **one** of its own, granted after the typing is
over. A single deadline across all three would let a slow write spend what the reply
needed — fritter would type the text and then be unable to say so, and a caller hearing
only that the connection closed sends the message twice. The sum is at most four seconds
against the **five** hands allows, so what hands hears is fritter's reason and not its own
timer. Widen any one without the others and a wedged session stops being able to say that
it is wedged.

`text` is typed literally. fritter does not decide what a leading `/` or `@` means to the
program underneath — that belongs to the caller, and in hands it is already settled
before anything reaches here. `submit` presses Enter afterwards.

`text` is characters and newlines, and a control byte in it is refused by name. A control
byte there is a keystroke in text's clothing: an `ESC` ends the bracketing early, so
everything after it is typed and submitted on its own, and a `0x03` is a Ctrl-C. Sent as
text they would turn one message into several, or into something nobody asked to send,
under an `ok`. Send a `key` request for a keystroke.

The keys are `escape`, `enter`, `ctrl_c`, `ctrl_u`, `up`, `down`, `tab` and `shift_tab`.
Which bytes each one is, is terminal knowledge, so that table lives here rather than in
the caller. `ctrl_u` is the one to reach for to empty the input box: `ctrl_c` empties it
too, but only on the first press — a second quits the session.

## Multi-line text

fritter watches the child's output for the sequence that turns bracketed paste on, and
wraps injected text to match. A newline inside the text is then a newline in the message
rather than the Enter that submits a half-written one.

This is read from the child rather than assumed, so a program that does not want
bracketing gets its text bare — and text holding a newline is refused outright when the
program has not asked for bracketing, because unbracketed it would arrive as several
separate submitted prompts. Answering `ok` to that would tell the caller one message was
sent where several were.

## Who owns the input line

If the person at the keyboard has characters in the box they have not sent, fritter
refuses to type text into it and says so. A half-written line is theirs until they send it
or throw it away.

This is the only fact about the input box fritter owns. Whether a session should be
written to at all — whether it is working, or sitting at a permission dialog — is the
caller's to decide, and hands decides it from state fritter cannot see. fritter's rule
covers the one thing the caller cannot know, because those keystrokes never reach it.

### Keys are never refused for it

A `key` request goes through whether the line is held or not. Text is the only thing that
interleaves: dropped into a half-written line it makes one prompt out of two people's
words, and afterwards nobody can pull them apart. A keystroke does exactly what it would
have done had the user pressed it, and the user sees the result.

It is also the way back. Enter, Ctrl-C and Ctrl-U are the keys that empty the box, so
refusing them would lock the door from the inside — a session whose line was held would be
reachable only by a human at the physical keyboard, which is the situation this program
exists to remove. A key that empties the box tells the line owner so, read by the same
parser that reads the keyboard, so there is one account of what those bytes mean and not
two.

### Why stdin has to be parsed and not scanned

The obvious way to follow the line is to watch stdin for Enter and Ctrl-C. It does not
work, and it fails silently. A terminal in raw mode also carries its answers to the
child's own questions: Claude Code turns on focus reporting and mouse reporting, so
`ESC [ O` arrives every time you tab away and mouse reports arrive as you move the
pointer. None of them holds an Enter, and all of them are full of printable bytes — read
as characters, `ESC [ <0;45;12M` is a pointer moving and ten keys pressed. Tab away
once and every send afterwards is refused into an empty box, for as long as the session
lives.

So `input.go` parses the stream. Escape sequences are skipped whole, by their shape rather
than their contents: CSI, SS3, the X10 mouse report that carries three raw bytes after
`ESC [ M` and so does not say where it ends, and the string sequences a terminal answers
longer questions with. A sequence cut in half by the end of a read is held over to the
next one. A paste the user made at their own keyboard is read as characters, newlines and
all, because that is exactly what the brackets around it say it is. What is left over is
typing.

The owner then counts characters rather than raising a flag, because a text box is a count
of characters. Backspacing back to an empty box hands the line back; a flag could never be
told that, and a fact that can only ever become true is not a fact about the box.

Enter, Ctrl-C and Ctrl-U empty the box; backspace takes one character out of it. Ctrl-W
takes the last word, and how many characters that is depends on what the word was, so it
is not counted: the line stays held until the user submits or cancels, or a `key` request
clears it. That is the safe direction to be wrong in — a refused write is loud and
recoverable, a write into a half-typed line is a garbled prompt nobody can attribute.

Escape is left alone, which is not the compromise an earlier version of this file claimed
it was: Claude Code 2.1.278 does not clear its input box on Escape. That was measured, not
assumed, and it is why the key is in the table but changes nothing here.

## What run promises, and what it does not

Termination signals are forwarded to the child, so the ordinary exit path runs and the
socket is removed and the terminal restored. Without that, a killed fritter would die with
its cleanup unrun, leaving a stale socket for the next caller to dial into nothing.

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

A write into the session gives up after a second. A pty in raw mode holds a kilobyte of
input and a write that fills it blocks until the child reads, which a running session does
at once and a stopped one never does. Waiting there with no bound hangs the request and
everything behind it — including the `ctrl_u` that was meant to be the way back. The write
cannot be taken back, so while one is outstanding every other is refused rather than queued
behind it, and it clears itself the moment the child starts reading again. What landed is
always reported: a write that failed partway says how many bytes reached the box, because
"nothing was typed" would send a caller to retype a message half of which is already there.

The goroutine reading your stdin outlives `run`. `main` hands it `os.Stdin`, and a read
already blocked there cannot be interrupted — `SetReadDeadline` answers *"file type does
not support deadline"* for a terminal — so joining on it would hang the exit rather than
hurry it. (Measured, because an earlier version of this file had it backwards: a pty slave
*does* take a deadline and *does* unblock a read in flight. That is the shape the tests run
in, not the shape a session runs in, and a contract that holds only under test is not one.)
The caller must not close stdin before its process ends, which for fritter is the next
statement in `main`. A guarantee that sometimes deadlocks is worse than one not made.

## Measured against Claude Code 2.1.278

- The child turns bracketed paste on, and text wrapped in it arrives as one multi-line
  message.
- Text followed immediately by Enter submits correctly. No delay is needed between them,
  so there is no timing bet in the send path.
- The child also turns on focus reporting and mouse reporting — `ESC [ ?1000h`,
  `?1002h`, `?1003h`, `?1006h` — which is why stdin carries far more than keypresses and
  why it is parsed rather than scanned. With those reports arriving on stdin, a send is
  still accepted; before this was parsed, one was enough to refuse every send afterwards.
- Ctrl-C and Ctrl-U each empty the input box. Escape does not touch it. Ctrl-W takes the
  last word. Backspacing to empty hands the line back, and a `ctrl_u` sent while the user
  holds the line clears it and leaves the session running.
- A two-line prompt sent with the display asleep arrived as one message and was answered.
- A pty in raw mode — which is what the child puts its side into — blocks a write at 1024
  bytes when nothing is reading. A cooked one takes 300 KB without blocking, which is why
  a test against a shell proves nothing here unless it makes the pty raw first.
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
