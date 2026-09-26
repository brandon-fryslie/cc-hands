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
written and tested against it; `send_draft` is what dials it. Nothing in fritter
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
{"pid":4242,"kind":"text","text":"fix the auth middleware","submit":true}
{"pid":4242,"kind":"key","key":"escape"}
```

`pid` is the process the caller means to type into, and a request naming any process but
the one fritter wrapped is refused before anything is typed. The address alone cannot say
which session it reaches: it is inherited, so a second session started from inside a
wrapped one — from its shell, or from a tmux server first started there — finds its
parent's address in its own environment. hands records each session's process with its
address, so it always has the pid to name. That is the process the hook runs under, so
wrap the program itself — `fritter -- claude` — and not a launcher that runs it as a
child: the launcher is what fritter wrapped, and every request naming the program under it
is refused.

The answer is `{"ok":true}` or `{"ok":false,"reason":"...","typed":"nothing"}`, where
`typed` is `nothing` when none of the request reached the program and `maybe` when some or
all of it may have. A reason always says what went wrong, because a write that was refused
and a write that landed must never look alike to the caller; `typed` says the part a caller
acts on, since resending after `nothing` is safe and resending after `maybe` can type the
text twice. When the text lands but the Enter after it does not, the reason says
so in those words — retyping text that is already sitting in the box would double it.

A caller has one second and 64 KB to get its request in. Past the size it is refused with
a reason that says so. Past the time, a connection that sent nothing is dropped, because a
client that connects and never writes would otherwise hold a goroutine for the rest of the
session; one that sent part of a line is answered for what it sent — a whole object without
its newline is carried out, and anything less is refused as unreadable. Both bounds are
generous for a local client sending one small object.

Each phase of an exchange is bounded on its own, and that is deliberate. Reading the
request gets **one** second, waiting for another write to finish gets **half** of one, the
request's writes into the session give up after **two** between them so their reason has
time to be written, and the reply gets **one** of its own, granted after the typing is over. A single deadline across all three would let a slow write spend what the reply
needed — fritter would type the text and then be unable to say so, and a caller hearing
only that the connection closed sends the message twice. The sum is at most four and a half
seconds against the **five** hands allows, so what hands hears is fritter's reason and not its own
timer. Widen any one without the others and a wedged session stops being able to say that
it is wedged.

`text` is typed into a box fritter has emptied first; see *Emptying the box*. It is typed literally. fritter does not decide what a leading `/` or `@` means to the
program underneath — that belongs to the caller, and in hands it is already settled
before anything reaches here. `submit` presses Enter afterwards — and a `submit` is refused
before anything is typed when the text ends where Claude Code would take that Enter as
something other than a send: after a backslash, which it turns into a newline, or on a
token that opens a completion list (the patterns are under *When Enter does not send*),
whose list takes the Enter. Typed into an empty box the cursor is at the end of the text,
so this is asked exactly. A space after the token
closes the list, because none of the patterns takes one.

`text` is characters and newlines, and a control byte in it is refused by name. A control
byte there is a keystroke in text's clothing: an `ESC` ends the bracketing early, so
everything after it is typed and submitted on its own, and a `0x03` is a Ctrl-C. Sent as
text they would turn one message into several, or into something nobody asked to send,
under an `ok`. Send a `key` request for a keystroke.

The keys are `escape`, `enter`, `ctrl_c`, `ctrl_u`, `up`, `down`, `tab` and `shift_tab`.
Which bytes each one is, is terminal knowledge, so that table lives here rather than in
the caller. A key goes in as it is, exactly as if the user had pressed it, and none of
them is needed to empty the box before text: fritter does that itself. `ctrl_c` interrupts
a working session; into an idle one it empties the box and arms a second press to quit.
`ctrl_u` kills back to the start of the line the cursor is on, and that is the line *as
displayed*: a prompt long enough to wrap loses one row and keeps the rest.

## Multi-line text

fritter watches the child's output for the sequence that turns bracketed paste on, and
wraps injected text to match. A newline inside the text is then a newline in the message
rather than the Enter that submits a half-written one.

This is read from the child rather than assumed, so a program that does not want
bracketing gets its text bare — and text holding a newline is refused outright when the
program has not asked for bracketing, because unbracketed it would arrive as several
separate submitted prompts. Answering `ok` to that would tell the caller one message was
sent where several were.

## Emptying the box

A `text` request types into an empty box, and fritter empties it itself, with Claude
Code's stash and without knowing anything about what the box or the stash held:

1. `a`, then Ctrl-S. Ctrl-S moves the box into the stash — every line of it, wherever the
   cursor is, in shell mode or out, with a completion list open or not — and leaves it
   empty in prompt mode. Into an *empty* box the same key puts the stash back instead, and
   the `a` is what makes sure the box is not empty when it lands.
2. `a`, then Ctrl-S again. The stash is now exactly `a`, and whatever it held is gone.
3. Ctrl-S into the empty box. The `a` comes back with the cursor after it, and the stash is
   empty.
4. Backspace, which takes the character before the cursor: that `a`.

The box is empty and so is the stash. What the person at the keyboard had half-written is
not kept, and neither is anything they had stashed: the stash is fritter's to use.

The stash has to end empty, not just the box. The session puts the stash back into the box
as it sends a prompt, and it does that some time after it reads the Enter — a stash
written a millisecond after the Enter was read put the text away unsent — so nothing can
safely follow the Enter to clean up. With nothing stashed, nothing comes back.

The `a` has to be a character the box keeps. A space typed into an empty box is dropped,
and a no-break space leaves the box reading as empty; either way the Ctrl-S after it
restored the stash under the text that came next.

Ctrl-C is not a way to empty the box. Into an idle session it empties it and arms a second
press to quit; into a working one it stops the work and leaves the box exactly as it was.
Which of the two a session is, is not something stdin says, and an earlier version of
fritter that tried to follow it from the keystrokes kept finding keys it had got wrong.

### One step per read

Each step — the four that empty the box, the text, and the marker that closes the paste
with the Enter after it — is written only once the session has read the one before.
Claude Code decides what a read means from the whole read: `a`, Ctrl-S and a long
bracketed paste arriving in one read were taken as one paste, the Ctrl-S stripped as an
invisible character, the `a` left at the front of the message, and the Enter held for
review. Short ones got through, which is what makes it dangerous. Written back to back,
the steps come out of one read whenever the session is a moment slow, so no spacing in
time keeps them apart.

What does is the pty's own count of the bytes waiting on the session's side of it. fritter
writes a step, waits for that count to reach zero, and writes the next; it waits the same
way before the first, so keys the user typed a moment earlier are read on their own too.
A terminal in canonical mode counts nothing until a whole line is in, so for a program that
reads lines the wait is no wait; Claude Code reads raw. The count is measured on macOS
only, and fritter builds nowhere else: Linux hands a pty write to the session's side on a
workqueue, so its count can read zero before the bytes are there to read.

The Enter is the one key that must not be read on its own. Read a millisecond after the
paste, the session took it before the paste was in the box and sent nothing. So it goes in
the same write as the marker that closes the paste, and the text is written before them:
however many reads a long text takes, the marker and the Enter arrive together.

### When Enter does not send

Two things in Claude Code take a Return and do something else with it. After a backslash
the backslash becomes a newline and everything stays in the box. And with a completion
list open the Return applies the highlighted entry and sends nothing. Whether a list is
open is decided by the token ending at the cursor, and the child's own patterns say which
tokens those are:

```js
@ /(^|[\s\u3002\u3001\uFF1F\uFF01])@([\p{L}\p{N}\p{M}_\-./\\()[\]~:]*|"[^"]*"?)$/u
# /(^|\s)#([a-z0-9][a-z0-9_-]*)$/
: /(^|\s):([a-z0-9_+-]{2,})$/
```

The `*` on the first is why a bare `@` counts. A slash command is not one of these — its
Return runs the command and empties the box. A `submit` whose text ends on either is
refused before anything is typed.

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

A request's writes give up two seconds after it starts writing, all of them together. Each one waits for the child to
read it, which a running session does at once and a stopped one never does. Waiting there
with no bound hangs the request and everything behind it. A write cannot be taken back, so
while one is outstanding nothing else writes: a request waits half a second for it and is
then refused rather than queued, and it clears itself the moment the child has read it.
The user's own keys wait for as long as it takes, because nothing may drop them, and they
wait only behind a request's writes, never inside one, so the user's typing cannot land
between a request's steps. What landed is
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

## Measured against Claude Code

On 2.1.283:

- `a` then Ctrl-S empties the box whatever is in it: a multi-line prompt, a `!` shell-mode
  prompt or a bare `!` (the box comes back in prompt mode), a prompt ending in an `@`
  token with its completion list open, and a prompt typed while a turn was running.
  Ctrl-S into an empty box puts the stash back.
- A space into an empty box is dropped, so a space then Ctrl-S restored the stash and the
  text after it was sent on the end of the user's old words.
- Submitting puts the stash back into the box ("Draft restored"), unless the prompt is a
  slash command, whose stash comes back when the command ends. It happens after the Enter is
  read, not as it is read: a Ctrl-S a millisecond after the Enter put the text away unsent.
- An Enter read a millisecond after the paste it follows sent nothing. Read together with
  the marker that closes the paste, it sent a short text and a 3 KB one.
- `a`, Ctrl-S, `a`, Ctrl-S, Ctrl-S, Backspace leaves the box and the stash empty, from a
  box holding a draft with the cursor inside it and a stash holding something else. Eight
  sends in a row from empty boxes, user drafts and unsent multi-line text each arrived as
  exactly the text sent and left the box empty.
- `a`, Ctrl-S and a bracketed paste of about 75 characters in one read were taken as one
  paste: the Ctrl-S removed as an invisible character, the `a` kept, the Enter held for
  review. At about 45 characters the same read did what it says. Read apart, every length
  did. A long bracketed paste followed by Enter in one read was sent.
- The pty's count of unread bytes on the session's side, `FIONREAD`, follows exactly what
  the session has not read yet, when the terminal is raw.
- The binary asks for bracketed paste, the kitty keyboard protocol (`>1u`) and
  modifyOtherKeys (`>4;2m`).

On 2.1.278:

- The child turns bracketed paste on, and text wrapped in it arrives as one multi-line
  message.
- One Ctrl-C into an idle session empties the input box however many lines are in it, and
  leaves the session running. A second press in a row quits it. Into a working session one
  Ctrl-C stops the work and leaves the box as it was.
- Ctrl-U does **not** empty the box. It kills back to the start of the line the cursor is
  on, and that is the *displayed* line. A box holding `aaa`, `bbb`, `ccc` took four presses
  and still had `aaa` in it; 250 characters typed into a 100-column terminal lost one
  wrapped row to a single press and kept 192.
- A Return pressed straight after a backslash does **not** submit. The backslash is
  replaced by a newline and the prompt stays in the box.
- A two-line prompt sent with the display asleep arrived as one message and was answered.
- A pty in raw mode — which is what the child puts its side into — blocks a write at 1022
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

The tests wrap a shell, or the test binary itself reading its terminal raw and recording
each read, rather than Claude Code, so they need no session and no network.
They use short temporary directories on purpose: `t.TempDir()` names the directory after
the test, which pushes the socket path past the 104-byte limit.
