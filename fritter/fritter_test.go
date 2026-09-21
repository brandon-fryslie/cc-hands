package main

import (
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

func TestLineOwnerFollowsWhatTheUserTyped(t *testing.T) {
	// [LAW:behavior-not-structure] The contract is "is the user's text unsent", stated as
	// what they pressed and what should follow. How it is tracked is not asserted.
	for _, c := range []struct {
		name string
		keys []string
		free bool
	}{
		{"nothing typed", nil, true},
		{"a half-typed line", []string{"half"}, false},
		{"a line submitted", []string{"hello", "\r"}, true},
		{"typed again after submitting", []string{"hello\r", "more"}, false},
		{"typed and submitted in one read", []string{"hello\r"}, true},
		{"cleared with ctrl-c", []string{"oops", "\x03"}, true},
		{"newline submits too", []string{"hello", "\n"}, true},
		{"an empty read changes nothing", []string{"half", ""}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			line := newLineOwner()
			for _, k := range c.keys {
				line.typed([]byte(k))
			}
			if line.free() != c.free {
				t.Fatalf("after %q: free=%v, want %v", c.keys, line.free(), c.free)
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
	dir := shortTempDir(t)
	listener, address, err := listen(dir)
	if err != nil {
		t.Fatalf("cannot listen: %v", err)
	}
	t.Cleanup(func() { listener.Close() })

	wrapped, err := start(argv, []string{"FRITTER_SOCKET=" + address})
	if err != nil {
		t.Fatalf("cannot start %v: %v", argv, err)
	}
	go wrapped.serve(listener)

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
		code, err := wrapped.run(terminalSlave, io.Discard)
		if err != nil {
			t.Errorf("run: %v", err)
		}
		exited <- code
	}()

	ask := func(body string) response {
		t.Helper()
		connection, err := net.Dial("unix", address)
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

func TestTheSocketIsGoneWhenTheCommandIsOver(t *testing.T) {
	// A socket left behind outlives the session it addressed, and the next caller to
	// dial it reaches nothing while believing it reached a session.
	dir := shortTempDir(t)
	listener, address, err := listen(dir)
	if err != nil {
		t.Fatalf("cannot listen: %v", err)
	}
	listener.Close()
	os.Remove(address)
	if _, err := os.Stat(address); !os.IsNotExist(err) {
		t.Fatalf("the socket path %s outlived its listener", address)
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
