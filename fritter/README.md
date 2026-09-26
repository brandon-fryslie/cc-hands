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

The answer is `{"ok":true}` or `{"ok":false,"reason":"..."}`. A reason always says what
went wrong, because a write that was refused and a write that landed must never look
alike to the caller. When the text lands but the Enter after it does not, the reason says
so in those words — retyping text that is already sitting in the box would double it.

A caller has one second and 64 KB to get its request in. Past the size it is refused with
a reason that says so. Past the time, a connection that sent nothing is dropped, because a
client that connects and never writes would otherwise hold a goroutine for the rest of the
session; one that sent part of a line is answered for what it sent — a whole object without
its newline is carried out, and anything less is refused as unreadable. Both bounds are
generous for a local client sending one small object.

Each phase of an exchange is bounded on its own, and that is deliberate. Reading the
request gets **one** second, waiting for another write to finish gets **half** of one, a
write into the session gives up after **one** so its reason has time to be written, and the
reply gets **one** of its own, granted after the typing is over. A single deadline across all three would let a slow write spend what the reply
needed — fritter would type the text and then be unable to say so, and a caller hearing
only that the connection closed sends the message twice. The sum is at most four and a half
seconds against the **five** hands allows, so what hands hears is fritter's reason and not its own
timer. Widen any one without the others and a wedged session stops being able to say that
it is wedged.

`text` is typed literally. fritter does not decide what a leading `/` or `@` means to the
program underneath — that belongs to the caller, and in hands it is already settled
before anything reaches here. `submit` presses Enter afterwards — and a `submit` is refused
before anything is typed when the text ends where Claude Code would take that Enter as
something other than a send: after a backslash, which it turns into a newline, or on a
token that opens a completion list (the patterns are under *Who owns the input line*),
whose list takes the Enter. Typed into an empty box the cursor is at the end of the text,
so this is asked exactly rather than read the way that holds. A space after the token
closes the list, because none of the patterns takes one.

`text` is characters and newlines, and a control byte in it is refused by name. A control
byte there is a keystroke in text's clothing: an `ESC` ends the bracketing early, so
everything after it is typed and submitted on its own, and a `0x03` is a Ctrl-C. Sent as
text they would turn one message into several, or into something nobody asked to send,
under an `ok`. Send a `key` request for a keystroke.

The keys are `escape`, `enter`, `ctrl_c`, `ctrl_u`, `up`, `down`, `tab` and `shift_tab`.
Which bytes each one is, is terminal knowledge, so that table lives here rather than in
the caller. `ctrl_c` is the one to reach for to empty the input box, and it is the only
key that empties it whatever is in it — but send it once, because a second press in a row
quits the session. `ctrl_u` kills back to the start of the line the cursor is on, and that
is the line *as displayed*: a prompt long enough to wrap loses one row and keeps the rest.
It is offered because a caller may want it, but it never hands the line back.

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

It is also the way back. Enter and Ctrl-C are the keys that empty the box, so
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

The owner then records one fact: something is in the box that was not seen to leave it.
A count was the obvious shape — a text box is a number of characters — and it was wrong,
because nothing arriving on stdin ever takes the number back down.

Backspace looked like the exception, and an earlier version of this file promised that
backspacing to an empty box hands the line back. It does not. The child reads the key as

```js
backspace(){if(this.isAtStart())return this;return this.left().modifyText(this)}
```

so a Backspace with the cursor at the start of the box removes nothing at all, and where
the cursor is is not something stdin says. Type `x`, press Ctrl-A, press Backspace, and a
counted box reaches zero with the character still in it. Hold the key down past the start
of a line — an autorepeat, not a corner — and it reaches zero with a whole line still in
it. A count that only ever rises is a flag that has learnt to add, so it is a flag.

Ctrl-C empties the box when the child is idle. That, and the characters themselves, is
the whole of what is known — and the qualification is load-bearing, because it is not yet
honoured. When the child is *working*, the same key interrupts the work and does not touch
the box at all: the user's half-typed next prompt is still sitting there afterwards, and
this reads the box as empty and hands the line back. Measured, and tracked as
`hands-harness-5nb.dh7`. Whether the child is working is not something stdin says — this
sees the Return that starts the work and never sees it end — so closing it needs a second
source of truth rather than another rule here, and that is a design change, not a parser
fix.

Everything else that is not a character is read as having changed the box by some amount
the bytes do not say, and that holds the line until the box is proved empty. Backspace
takes one character or none; Ctrl-W takes a word; Ctrl-U takes back to the start of the
displayed line, so a wrapped prompt loses a row and keeps the rest; Ctrl-Y pastes back
whatever was last killed; Tab completes a path in; Up and Down pull a whole previous
prompt into a box nothing was typed into. Four
chords are listed as leaving the box alone — Ctrl-A, Ctrl-B, Ctrl-E, Ctrl-F — along with
the sequences that are the terminal answering a question, and the cursor keys that only
move sideways.

The reason the list is that short is that almost every edit is defined against something
invisible from here: the cursor, or the width of the terminal. Ctrl-U is bounded by both.
Nothing on stdin says where the cursor is, so anything measured from it is unknowable, and
unknowable holds the line.

Listing the harmless ones and holding the line for everything else is the only arrangement
that does not need the list to be complete, and it will never be complete: the ways to
edit a text box belong to the child and change when the child changes. Successive reviews
of this file each found another key that had been assumed harmless and was not. That is
the safe direction to be wrong in — a refused write is loud and recoverable, a write into
a half-typed line is a garbled prompt nobody can attribute.

Enter usually empties the box, and the exceptions are the two ways a multi-line prompt
gets written by hand — so they are not corners, and read as submits each one empties a
count that is not empty and lets hands write into the middle of somebody's sentence.

Claude Code reads a Return that follows a backslash as *keep typing*: the backslash
becomes a newline and everything already typed stays where it is. And Ctrl-J, which the
child offers in its own footer, is a different key from Return — it arrives as a bare
`0x0A` where Return arrives as `0x0D`, and it puts a newline in the box and sends nothing.
Both were measured against the running program by typing a prompt, pressing the key, and
watching the words stay.

The third is a completion list. With one open the child calls `preventDefault()` on the
Return and applies the highlighted entry instead, which leaves the box *longer* than it
was and still unsent. Whether a list is open is decided by the token ending at the cursor,
and the child's own patterns say which tokens those are:

```js
@ /(^|[\s\u3002\u3001\uFF1F\uFF01])@([\p{L}\p{N}\p{M}_\-./\\()[\]~:]*|"[^"]*"?)$/u
# /(^|\s)#([a-z0-9][a-z0-9_-]*)$/
: /(^|\s):([a-z0-9_+-]{2,})$/
```

An `@` naming a file is how a prompt points at code, and a directory keeps the list open
for the press after, so a user tabbing a path down with Return is ordinary use. The `*` on
the first pattern is why a bare `@` counts: a cursor sitting straight after one already
opens the list on every file there is. A slash command is not one of these — its Return
runs the command and empties the box.

Those two rules are why the owner keeps the end of the line rather than only a flag.
Sixteen characters of the line's end are remembered. Nothing comes off that end, because
nothing on stdin says how much came off the box, so what is remembered is a superset of
what the line really ends with — and a superset can only hold a Return that would have
sent, never free one that would not.

Where the cursor is, is not tracked — nothing on stdin says. So a backslash anywhere in
those sixteen characters holds the Return after it, and so does any cursor position among
them that would have opened a completion list. What that leaves open is a token further
back in the line than is remembered with the cursor parked inside it; and in the other
direction, a line held that was really sent. That one clears when the user types sixteen
more characters, or at once if hands sends a `ctrl_c` — a key request is never refused,
so a held line can always be handed back.

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
everything behind it — including the `ctrl_c` that was meant to be the way back. The write
cannot be taken back, so while one is outstanding nothing else writes: a request waits half
a second for it and is then refused rather than queued, and it clears itself the moment the
child starts reading again. The user's own keys wait for as long as it takes, because
nothing may drop them, and they wait only behind a request's writes, never inside one, so a
request's check and the text it types cannot have the user's typing land between them. What landed is
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
- One Ctrl-C empties the input box however many lines are in it, and leaves the session
  running. A second press in a row quits it.
- Ctrl-U does **not** empty the box. It kills back to the start of the line the cursor is
  on, and that is the *displayed* line. A box holding `aaa`, `bbb`, `ccc` took four presses
  and still had `aaa` in it; 250 characters typed into a 100-column terminal lost one
  wrapped row to a single press and kept 192. On a short one-line box it does clear it,
  which is why an earlier version of this file said it cleared the box.
- A Return continues the line when the character *before the cursor* is a backslash, not
  only when the line ends in one. Typing `ab\c`, pressing Left once and pressing Return
  left `ab` and `c` in the box.
- Escape does not touch the box. Ctrl-W takes the last word. Backspace takes one character
  or none, so backspacing to what looks like an empty box does not hand the line back.
- Up pulls the previous prompt into an empty box. Nothing was typed and the box filled,
  which is why a history key holds the line.
- A Return pressed straight after a backslash does **not** submit. The backslash is
  replaced by a newline and the prompt stays in the box, which is how a multi-line prompt
  is written by hand. Measured by typing `please fix the auth bug in \` and pressing
  Return: the box kept the words and grew a line.
- Ctrl-J does not submit either. It is a bare `0x0A`, a different key from Return's
  `0x0D`, and it is the multi-line prompt for any terminal that cannot send Shift-Enter.
  Measured the same way: the prompt stayed and the box grew a line.
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
