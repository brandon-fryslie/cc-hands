# fritter

fritter runs an interactive terminal program and adds a second input channel to it. The
program behaves exactly as it does in your own terminal, with the same interface, width,
signals, and exit code. At the same time, fritter listens on a unix socket. Input sent to
that socket is typed into the program's input, as if someone had typed it at the
keyboard.

```
fritter [--socket-dir DIR] [--tap UPSTREAM --tap-ca VARIABLE --tap-to SOCKET] -- COMMAND [ARGS...]
```

fritter was built for hands. hands' client for it, `hands.sessions.typing`, is written and
tested against it, and hands' `send_draft`, `send_command` and `interrupt_session` connect
to it. fritter itself has no knowledge of hands; Claude Code is the first program it
wraps.

## Why a pseudo-terminal and not a pipe

Claude Code checks whether its input is a terminal. If it is not, Claude Code runs in a
different, non-interactive mode. A wrapper that uses pipes would therefore control a
different program from the one you want to reach. For this reason, fritter allocates a
pty, runs the command on the slave side, and keeps the master side.

Using a pty also lets fritter work while the screen is off. It does not need to focus a
window, synthesize keystrokes, or request Accessibility or Input Monitoring permission. It
only writes bytes to a file descriptor. It works over SSH and with the laptop lid closed.

## Finding the socket

fritter chooses the socket path and passes it to the child in `FRITTER_SOCKET`. Every
process the child starts inherits that variable. The variable is the only connection
between fritter and the program that controls it.

hands uses this variable to find a session. Claude Code runs the hands hook as a child of
the session process. The hook reads `FRITTER_SOCKET` from its own environment, and the
address is written to the session's membership file. Neither side needs to agree on a
filename or derive one from a pid.

The socket is in its own directory, which has mode 0700, and the socket itself has mode
0600. Both are required. The default parent directory is the value of `TMPDIR`. For a
login shell this directory is private, but under launchd and cron it is the
world-writable `/tmp`. Any local user who can connect to the socket can type commands such
as `!rm -rf ~` into the program. fritter removes the directory when it exits.

macOS limits a unix socket path to about 104 bytes. fritter checks the length when it
chooses the path and reports the problem, instead of letting `bind` fail with an error
that does not identify the cause.

## The protocol

Each connection carries one newline-terminated JSON object, and fritter replies with one
JSON object.

```
{"pid":4242,"kind":"text","text":"fix the auth middleware"}
{"pid":4242,"kind":"command","command":"/btw","text":"what did the last test say?"}
{"pid":4242,"kind":"key","key":"escape"}
```

`pid` is the process the caller intends to type into. fritter refuses a request that names
any process other than the one it wrapped. The socket address alone does not identify
which session it reaches, because the address is inherited: a second session started from
inside a wrapped session finds its parent's address in its own environment. For this
reason, wrap the program itself (`fritter -- claude`), not a launcher that runs it as a
child. hands' `claude` shim, which `hands install-fritter` writes, does this for every
interactive session.

fritter types a `text` request the way someone types a message at the keyboard: the text,
then Return. It sends this to the child in a single write. fritter does not interpret a
leading `/` or `@` for the program; the caller decides what it means. In hands, this is
decided before the request reaches fritter.

A `command` request contains a command and its text. fritter types the command as
keystrokes, then a space, then pastes the text, then sends Return, all in a single write.
Only the text is pasted, because Claude Code collapses a long paste into a placeholder,
`[Pasted text #1 +80 lines]`. A command inside the collapsed paste is no longer at the
start of the input, so Claude Code reads the entire input as a prompt. A command is one
word. If there is no text, fritter types the command by itself, then Return.

A `text` value may contain only characters and newlines. fritter refuses a request that
contains a control byte, and the error names the byte: an `ESC` would end the bracketed
paste early, and `0x03` is Ctrl-C. To send a keystroke, use a `key` request. The
supported keys are `escape`, `enter`, `ctrl_c`, `ctrl_u`, `up`, `down`, `tab` and
`shift_tab`. Each key is typed exactly as if the user had pressed it.

A `pasting` request, `{"pid":4242,"kind":"pasting"}`, does not type anything. fritter
returns ok if the child has enabled bracketed paste, and refuses the request if it has
not. Claude Code enables bracketed paste once its input is ready. A caller that has just
started Claude Code and is about to type the first prompt waits for this request to
succeed, instead of waiting for an estimated length of time.

fritter returns `{"ok":true}` after the write completes, or `{"ok":false,"reason":"..."}`
if the request was refused or the write failed. Writes to the child happen one at a time,
for both the user's keystrokes and socket requests, so one write never lands inside
another. After it writes an `escape`, fritter writes nothing else to the input for 100ms,
and sends its reply after that delay. A terminal reads an `ESC` that is closely followed by another
byte as a single key combination, so Escape and Ctrl-C sent together would reach Claude
Code as Alt+Ctrl-C.

A caller must send its request within one second, and the request must not exceed 64 KB.

## The tap

With `--tap UPSTREAM --tap-ca VARIABLE --tap-to SOCKET`, fritter acts as the child's HTTP
proxy. It sets `HTTPS_PROXY`, `https_proxy`, `HTTP_PROXY` and `http_proxy` in the child's
environment to its own address, and sets `NO_PROXY` and `no_proxy` to empty, so every
connection the child makes goes to fritter. fritter then connects to the hosts the child
requested, as the child would have, using fritter's own environment (including any proxy
fritter was given; see below). The child
continues to use UPSTREAM as its server address. From the child's point of view, the
server has not changed, so it keeps any data it stores for that server.

fritter answers connections that the child opens to UPSTREAM's host itself. It presents a
certificate for that host, signed by a certificate authority that fritter generates at
startup and keeps only in memory. `VARIABLE` is the name of the variable that specifies
which file holds the certificates the child trusts in addition to its built-in ones
(`NODE_EXTRA_CA_CERTS` for Node). fritter sets that variable to a file in its socket
directory that contains the original file's certificates plus fritter's certificate
authority. fritter forwards each request on that connection to UPSTREAM unchanged, and
streams the response back as it arrives. An `http` UPSTREAM is tapped the same way, using
the child's plain HTTP requests to it. All other connections pass through fritter without
being decrypted. If fritter itself was given a proxy, those connections go through that
proxy.

fritter sends a copy of each exchange with UPSTREAM to `SOCKET`, using one connection per
exchange, as JSON lines in the order the events occurred:

```
{"kind":"request","at":…,"method":"POST","path":"/v1/messages","headers":[[name,value]…],"body":"<base64>","lost":0}
{"kind":"response","at":…,"status":200,"headers":[…]}
{"kind":"bytes","at":…,"bytes":"<base64>"}          (one per chunk, as it came)
{"kind":"end","at":…,"error":""}                     (or "unreached", with its error, in place of the reply)
```

`at` is the time in seconds since the epoch. Headers that contain credentials are never
copied. Bodies are copied in full, and the listener decides which parts to keep. The
child never waits for the copy. If no listener is connected, or the listener stops
reading, the exchange is not affected. If a copy's request was not delivered to the
listener, it is counted in the `lost` field of the next copy whose request is delivered.
If a copy is interrupted after its request was delivered, it ends without its remaining
lines, and the listener sees it end that way. When the child exits, fritter gives the
copies still in progress a short time to finish before it exits.

The proxy and the certificate file stop existing when fritter exits, but processes the
child starts inherit the variables that point to them. For this reason, fritter publishes
its proxy address in `FRITTER_TAP`, and the previous value of each variable it set in
`FRITTER_OUTER_<name>`. A program that the child runs can use these to detect that a proxy
setting still points to the tap, and to restore the previous values. The caller must know
which variable a program reads its certificates from; hands' `claude` shim passes
`NODE_EXTRA_CA_CERTS`.

## Multi-line text

fritter monitors the child's output for the escape sequence that enables bracketed paste,
and when it is enabled, wraps injected text in bracketed-paste markers. A newline in the
text is then a newline in the message, not a Return that submits part of it. If the
program has not enabled bracketed paste, fritter refuses text that contains a newline,
because it would arrive as several separate prompts.

## What run promises, and what it does not

fritter forwards termination signals to the child, so the normal exit path runs: the
socket is removed and the terminal is restored. Without forwarding, a killed fritter would
exit without running its cleanup, and would leave a stale socket that the next caller
would connect to with no process behind it.

Closing the terminal ends the session. The terminal's hangup reaches fritter as SIGHUP,
which fritter forwards like any other termination signal. fritter reads the child's output
from the pty for as long as the child writes, whether or not the terminal is still there
to display it. If nobody reads a child's output, the child blocks while exiting (Claude
Code waits in `tcsetattr` for its output to drain). It would then keep running, with
fritter waiting on it, after the terminal window had closed.

fritter exits with the child's exit code. If a signal killed the child, fritter reports
128 plus the signal number, as a shell does; for example, SIGTERM gives 143. A child
killed by a signal has no exit code of its own. Passing on Go's -1 would make fritter exit
with code 255, and a caller could not distinguish that from a program that actually exited
with code 255.

`run` waits for the child's final output before it returns. `cmd.Wait` returns when the
child is reaped, and there is no ordering between that event and the end of copying the
child's output. If `run` returned at that point, the caller would receive a finished
session whose last output is still being written, and the caller's next action is to exit.
With a slow stdout, the lost output is easy to see. With a fast one, it is a race
condition that almost always succeeds, which is the worse kind. The wait times out
after two seconds, because a process the child left running can keep the pty open, and an
unbounded wait would keep fritter running after its session ended. If the timeout is
reached, output was lost, and fritter reports it.

The goroutine that reads your stdin keeps running after `run` returns. `main` passes it
`os.Stdin`, and a read that is already blocked there cannot be interrupted: for a
terminal, `SetReadDeadline` returns *"file type does not support deadline"*. Waiting for
the goroutine to finish would therefore make the exit hang instead of making it faster.
(This was measured, because an earlier version of this file stated the opposite: a pty
slave *does* accept a deadline and *does* unblock a read in progress. However, that is the
setup the tests run in, not the setup a real session runs in, and a guarantee that holds
only under test is not a guarantee.) The caller must not close stdin before its process
exits. For fritter, the process exits in the next statement in `main`. A guarantee that
sometimes deadlocks is worse than no guarantee.

## Measured against Claude Code

On 2.1.283:

- When a bracketed paste and the Return after it are read together, the message is sent.
  This is true for both a short text and a 3 KB text. A Return read a moment after the
  paste, separately from it, sent nothing. This is why the two are sent in a single write.
- When they are read together in this way, text that ends with an `@` or `#` token is sent
  unchanged: no completion list opens and consumes the Return. Text that ends with a
  backslash is not sent. The Return becomes a newline, and the next text typed is
  submitted joined to it.
- Text that starts with a space followed by `/`, `@` or `!` is sent as plain text, and the
  transcript records it with the leading space.
- The binary enables bracketed paste.
- Killing the tmux session that a wrapped session runs in, or sending fritter SIGHUP or
  SIGTERM, ends fritter, claude and the processes claude started within one second, and
  removes the socket directory.

On 2.1.278:

- Text wrapped in bracketed paste arrives as one multi-line message.
- A two-line prompt sent while the display was asleep arrived as one message and received
  a response.
- A process that the child starts can read `FRITTER_SOCKET`.
- A permission dialog and the workspace-trust dialog each consume a paste and treat the
  Return as their answer. Do not send text to a session that is showing a dialog.

## Building and testing

```
cd fritter
go test -race ./...
go build -o fritter .
```

The tests wrap a shell, or the test binary itself reading its terminal in raw mode and
recording each read, instead of Claude Code. As a result, they need no session and no
network access. They use short temporary directories on purpose: `t.TempDir()` names the
directory after the test, which makes the socket path longer than the 104-byte limit.
