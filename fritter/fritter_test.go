package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/creack/pty"
)

func TestTheLineIsHeldByWhatTheUserTypedAndNothingElse(t *testing.T) {
	// [LAW:behavior-not-structure] The contract is "has the user got unsent characters in
	// the box", stated as what arrived on stdin and what should follow. How it is tracked
	// is not asserted.
	//
	// Everything below the first group is a terminal answering the child rather than a
	// person typing. Claude Code turns focus reporting and mouse reporting on - the
	// captured pty output from this session shows it sending \x1b[?1000h \x1b[?1002h
	// \x1b[?1003h \x1b[?1006h - so these arrive on stdin during any ordinary session, and
	// they are full of printable bytes. Counted as typing they hold the line for good:
	// tab away from the terminal once and every send afterwards is refused into an empty
	// box. That is what these cases exist to prevent.
	for _, c := range []struct {
		name  string
		reads []string
		free  bool
	}{
		{"nothing typed", nil, true},
		{"a half-typed line", []string{"half"}, false},
		{"a line submitted", []string{"hello", "\r"}, true},
		{"typed again after submitting", []string{"hello\r", "more"}, false},
		{"typed and submitted in one read", []string{"hello\r"}, true},
		{"cleared with ctrl-c", []string{"oops", "\x03"}, true},
		{"newline submits too", []string{"hello", "\n"}, true},
		{"an empty read changes nothing", []string{"half", ""}, false},
		{"backspaced back to empty", []string{"abc", "\x7f\x7f\x7f"}, true},
		{"backspaced most of the way", []string{"abc", "\x7f\x7f"}, false},
		{"backspaced past empty", []string{"a", "\x7f\x7f\x7f"}, true},

		{"the window losing focus", []string{"\x1b[O"}, true},
		{"the window gaining focus", []string{"\x1b[I"}, true},
		{"focus lost while a line is held", []string{"half", "\x1b[O"}, false},
		{"an sgr mouse report", []string{"\x1b[<35;89;12M"}, true},
		{"an x10 mouse report whose bytes are printable", []string{"\x1b[M !!"}, true},
		{"a cursor position report", []string{"\x1b[24;80R"}, true},
		{"a device attributes report", []string{"\x1b[?1;2c"}, true},
		{"a colour query answered", []string{"\x1b]11;rgb:1b1b/1b1b/1b1b\x07"}, true},
		{"a report split across two reads", []string{"\x1b[<35;8", "9;12M"}, true},
		{"an escape split from its sequence", []string{"\x1b", "[A"}, true},
		{"arrow keys move without typing", []string{"\x1b[A", "\x1b[B", "\x1bOC"}, true},
		{"escape alone is not a character", []string{"\x1b"}, true},

		{"a pasted line is characters in the box", []string{"\x1b[200~hello\nworld\x1b[201~"}, false},
		{"a newline inside a paste does not submit", []string{"\x1b[200~a\nb\x1b[201~"}, false},
		{"a paste spanning reads", []string{"\x1b[200~a\nb", "c\x1b[201~"}, false},
		{"a paste whose end marker straddles a read", []string{"\x1b[200~ab\x1b[20", "1~"}, false},
		{"a paste then submitted", []string{"\x1b[200~a\nb\x1b[201~", "\r"}, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			line := newLineOwner()
			for _, r := range c.reads {
				line.typed([]byte(r))
			}
			if line.free() != c.free {
				t.Fatalf("after %q: free=%v, want %v", c.reads, line.free(), c.free)
			}
		})
	}
}

func TestAChordFritterSendsMeansWhatTheSameChordMeansTyped(t *testing.T) {
	// The line has to be recoverable without a human at the keyboard, and these are the
	// keys that recover it. If fritter's own Enter or Ctrl-C did not count, a held line
	// would stay held after the very request sent to clear it.
	for _, c := range []struct {
		name string
		key  string
		free bool
	}{
		{"enter submits the held line", "enter", true},
		{"ctrl-c throws it away", "ctrl_c", true},
		{"an arrow key leaves it where it is", "up", false},
		{"tab leaves it where it is", "tab", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			line := newLineOwner()
			line.typed([]byte("theirs"))
			line.sent(keystrokes[c.key])
			if line.free() != c.free {
				t.Fatalf("after sending %s: free=%v, want %v", c.key, line.free(), c.free)
			}
		})
	}
}

func TestPasteModeReadsWhatTheChildAnnounced(t *testing.T) {
	mode := newPasteMode()
	if mode.enabled() {
		t.Fatal("bracketed paste must be off until the child asks for it")
	}
	mode.Write([]byte("some output\x1b[?2004hmore"))
	if !mode.enabled() {
		t.Fatal("the child turned bracketed paste on and it was not noticed")
	}
	mode.Write([]byte("\x1b[?2004l"))
	if mode.enabled() {
		t.Fatal("the child turned bracketed paste off and it was not noticed")
	}
}

func TestPasteModeSurvivesASequenceSplitAcrossReads(t *testing.T) {
	// A pty read can end anywhere, including the middle of the sequence that announces
	// the mode. Missing it would mean injecting bare newlines into a session that would
	// have taken a paste, which submits a half-written message.
	mode := newPasteMode()
	mode.Write([]byte("output\x1b[?20"))
	mode.Write([]byte("04h"))
	if !mode.enabled() {
		t.Fatal("a split mode sequence was missed")
	}
}

func TestEncodeBracketsOnlyWhenTheChildAcceptsIt(t *testing.T) {
	mode := newPasteMode()
	if got := string(mode.encode("a\nb")); got != "a\nb" {
		t.Fatalf("with paste off, text must go as it is, got %q", got)
	}
	mode.Write([]byte("\x1b[?2004h"))
	if got := string(mode.encode("a\nb")); got != "\x1b[200~a\nb\x1b[201~" {
		t.Fatalf("with paste on, text must be bracketed, got %q", got)
	}
}

// wrap runs argv under fritter on a pty of its own and returns a way to ask it things.
func wrap(t *testing.T, argv ...string) (*Wrapped, func(string) response, chan int) {
	t.Helper()
	return wrapOnto(t, io.Discard, argv...)
}

func wrapOnto(t *testing.T, stdout io.Writer, argv ...string) (*Wrapped, func(string) response, chan int) {
	t.Helper()
	socket, err := listen(shortTempDir(t))
	if err != nil {
		t.Fatalf("cannot listen: %v", err)
	}
	t.Cleanup(socket.close)

	wrapped, err := start(argv, []string{"FRITTER_SOCKET=" + socket.address})
	if err != nil {
		t.Fatalf("cannot start %v: %v", argv, err)
	}
	go wrapped.serve(socket.listener)

	// fritter puts its own terminal into raw mode, so the test gives it a real one.
	terminal, terminalSlave, err := pty.Open()
	if err != nil {
		t.Fatalf("cannot open a terminal for the test: %v", err)
	}
	released := make(chan struct{})
	// Closed only once run has returned and let go of the terminal, which is what the
	// caller of run is required to do and what fritter's own main does by exiting.
	t.Cleanup(func() {
		select {
		case <-released:
		case <-time.After(5 * time.Second):
			t.Error("run did not return; the terminal is being closed under it")
		}
		terminal.Close()
		terminalSlave.Close()
	})

	exited := make(chan int, 1)
	go func() {
		defer close(released)
		// The child's output goes nowhere the test reads back into fritter's stdin: a
		// real terminal does not hand back what was printed to it, and a harness that
		// does would count the child's own output as the user typing.
		code, err := wrapped.run(terminalSlave, stdout)
		if err != nil {
			t.Errorf("run: %v", err)
		}
		exited <- code
	}()

	ask := func(body string) response {
		t.Helper()
		connection, err := net.Dial("unix", socket.address)
		if err != nil {
			t.Fatalf("cannot dial the control socket: %v", err)
		}
		defer connection.Close()
		if _, err := connection.Write([]byte(body + "\n")); err != nil {
			t.Fatalf("cannot ask: %v", err)
		}
		var answer response
		if err := json.NewDecoder(connection).Decode(&answer); err != nil {
			t.Fatalf("cannot read the reply: %v", err)
		}
		return answer
	}
	return wrapped, ask, exited
}

func TestInjectedTextReachesTheChildAndTheExitCodeIsTheChildsOwn(t *testing.T) {
	// The child reads one line and exits with 3, so the test proves both that the text
	// arrived and that fritter did not invent an exit code of its own.
	_, ask, exited := wrap(t, "sh", "-c", "read line; test \"$line\" = deliberate && exit 3 || exit 9")

	if answer := ask(`{"kind":"text","text":"deliberate","submit":true}`); !answer.OK {
		t.Fatalf("the write was refused: %s", answer.Reason)
	}
	select {
	case code := <-exited:
		if code != 3 {
			t.Fatalf("exit code %d: the child did not receive %q, or fritter rewrote its code", code, "deliberate")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited; the text probably never arrived")
	}
}

func TestAChildKilledByASignalIsReportedAsOne(t *testing.T) {
	// A signalled child has no exit code of its own. Passing on the -1 that Go reports
	// would exit 255, which is a code a program can genuinely exit with, so a caller
	// could not tell "exited 255" from "was killed". 128 plus the signal is what shells
	// report and what everything reading exit codes already understands.
	_, _, exited := wrap(t, "sh", "-c", "kill -TERM $$")
	select {
	case code := <-exited:
		if code != 128+int(syscall.SIGTERM) {
			t.Fatalf("exit code %d, want %d for a child killed by SIGTERM", code, 128+int(syscall.SIGTERM))
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited")
	}
}

func TestTheChildsLastOutputIsWrittenBeforeRunReturns(t *testing.T) {
	// cmd.Wait returns when the child is reaped, which is before the last of what it
	// printed has come back through the pty. A run that returned there would let the
	// caller exit with the output still in flight, so the shortest possible session -
	// print one word and quit - would print nothing.
	var printed bytes.Buffer
	_, _, exited := wrapOnto(t, &printed, "sh", "-c", "echo done")
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited")
	}
	if !strings.Contains(printed.String(), "done") {
		t.Fatalf("the child printed \"done\" and run returned with %q", printed.String())
	}
}

func TestTheUserKeepsTheirOwnHalfTypedLine(t *testing.T) {
	// The child takes two lines: the one the user types and the one fritter injects once
	// they have submitted it. Submitting is what hands the line back here; that Ctrl-C
	// does too is asserted in the lineOwner table, where it needs no live child and no
	// cooked-terminal signal semantics to be true.
	wrapped, ask, exited := wrap(t, "sh", "-c", "IFS= read -r a; IFS= read -r b; exit 0")

	// The user types without submitting, exactly as Wrapped.Write records it.
	if _, err := wrapped.Write([]byte("mine")); err != nil {
		t.Fatalf("cannot type: %v", err)
	}
	answer := ask(`{"kind":"text","text":"intruder","submit":true}`)
	if answer.OK {
		t.Fatal("a write landed in the middle of the user's line")
	}
	if !strings.Contains(answer.Reason, "unsent text") {
		t.Fatalf("the refusal must say why, got %q", answer.Reason)
	}

	// The user submits, so the box is empty and the line is fritter's to write into.
	if _, err := wrapped.Write([]byte("\r")); err != nil {
		t.Fatalf("cannot submit: %v", err)
	}
	if answer := ask(`{"kind":"text","text":"now ok","submit":true}`); !answer.OK {
		t.Fatalf("the line was handed back but the write was still refused: %s", answer.Reason)
	}
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never took the injected line")
	}
}

func TestAKeyGoesThroughAHeldLineAndGivesItBack(t *testing.T) {
	// Text yields to the person at the keyboard; a key does not. Gating keys too would
	// lock the door from the inside, because Enter and Ctrl-C are the keys that free a
	// line: a session whose line was held would have no way back but a human at the
	// physical keyboard, which is the case fritter exists to avoid.
	wrapped, ask, exited := wrap(t, "sh", "-c", "IFS= read -r a; IFS= read -r b; exit 0")

	if _, err := wrapped.Write([]byte("theirs")); err != nil {
		t.Fatalf("cannot type: %v", err)
	}
	if answer := ask(`{"kind":"text","text":"intruder","submit":true}`); answer.OK {
		t.Fatal("text landed in the middle of the user's line")
	}
	if answer := ask(`{"kind":"key","key":"enter"}`); !answer.OK {
		t.Fatalf("a key was refused into a held line, so nothing can ever free it: %s", answer.Reason)
	}
	if answer := ask(`{"kind":"text","text":"now ok","submit":true}`); !answer.OK {
		t.Fatalf("the key submitted the line but fritter still thinks it is held: %s", answer.Reason)
	}
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never took both lines")
	}
}

func TestMultiLineTextIsRefusedWhenTheSessionWillNotBracketIt(t *testing.T) {
	// A shell never turns bracketed paste on, so a newline here is an Enter. Answering ok
	// would tell hands one message was sent where several separate prompts were.
	_, ask, _ := wrap(t, "sh", "-c", "IFS= read -r a; exit 0")

	answer := ask(`{"kind":"text","text":"first\nsecond","submit":true}`)
	if answer.OK {
		t.Fatal("a multi-line draft was accepted into a session that cannot take one whole")
	}
	if !strings.Contains(answer.Reason, "bracketed paste") {
		t.Fatalf("the refusal must say why, got %q", answer.Reason)
	}
	if answer := ask(`{"kind":"text","text":"one line","submit":true}`); !answer.OK {
		t.Fatalf("the child could not be let go: %s", answer.Reason)
	}
}

func TestRequestsThatNameNothingRealAreRefusedWithAReason(t *testing.T) {
	_, ask, _ := wrap(t, "sh", "-c", "read line; exit 0")

	for _, c := range []struct{ name, body, says string }{
		{"an unknown key", `{"kind":"key","key":"f13"}`, "no key named"},
		{"an unknown kind", `{"kind":"shout","text":"hi"}`, "no request kind named"},
		{"not json at all", `nonsense`, "cannot read the request"},
	} {
		t.Run(c.name, func(t *testing.T) {
			answer := ask(c.body)
			if answer.OK {
				t.Fatal("this should not have been accepted")
			}
			if !strings.Contains(answer.Reason, c.says) {
				t.Fatalf("the reason must say %q, got %q", c.says, answer.Reason)
			}
		})
	}
	// Every request above was refused, so the child is still waiting on its line. Give it
	// one, or the wrapper never finishes and the test tears its terminal down underneath.
	if answer := ask(`{"kind":"text","text":"done","submit":true}`); !answer.OK {
		t.Fatalf("the child could not be let go: %s", answer.Reason)
	}
}

func TestTheControlSocketIsTheUsersAlone(t *testing.T) {
	// The default parent is whatever TMPDIR names, which is private for a login shell and
	// is the world-writable /tmp under launchd and cron. A socket anyone may dial is a
	// socket anyone may type `!rm -rf ~` into, so the modes are the test.
	socket, err := listen(shortTempDir(t))
	if err != nil {
		t.Fatalf("cannot listen: %v", err)
	}
	defer socket.close()

	for _, c := range []struct {
		what string
		path string
		want os.FileMode
	}{
		{"the socket", socket.address, 0o600},
		{"its directory", socket.dir, 0o700},
	} {
		info, err := os.Stat(c.path)
		if err != nil {
			t.Fatalf("cannot stat %s: %v", c.what, err)
		}
		if got := info.Mode().Perm(); got != c.want {
			t.Errorf("%s is %o, want %o: another local user can reach this session", c.what, got, c.want)
		}
	}
}

func TestTheSocketIsGoneWhenTheCommandIsOver(t *testing.T) {
	// A socket left behind outlives the session it addressed, and the next caller to
	// dial it reaches nothing while believing it reached a session.
	socket, err := listen(shortTempDir(t))
	if err != nil {
		t.Fatalf("cannot listen: %v", err)
	}
	socket.close()
	if _, err := os.Stat(socket.address); !os.IsNotExist(err) {
		t.Fatalf("the socket path %s outlived its listener", socket.address)
	}
	if _, err := os.Stat(socket.dir); !os.IsNotExist(err) {
		t.Fatalf("the socket directory %s outlived its listener", socket.dir)
	}
}

// shortTempDir gives a directory whose paths fit in a unix socket address.
//
// t.TempDir() names the directory after the test, which on macOS puts a socket path well
// past the 104 bytes the kernel allows. The length limit is real and fritter reports it;
// this is how the tests stay inside it.
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("/tmp", "fritter-test-")
	if err != nil {
		t.Fatalf("cannot make a temp dir: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}
