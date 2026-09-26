package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

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

func TestEncodeBracketsOnlyWhenTheChildAcceptsItAndSaysWhichItDid(t *testing.T) {
	// The caller needs the same reading of the mode that the bytes were made with. Asked
	// twice - once to decide whether multi-line text is safe, once to encode it - the
	// child can turn bracketing off in between, and a message accepted as one paste goes
	// as several submitted prompts under an ok.
	mode := newPasteMode()
	body, end, bracketed := mode.encode("a\nb")
	if string(body)+string(end) != "a\nb" || bracketed {
		t.Fatalf("with paste off, text must go as it is and say so, got %q%q bracketed=%v", body, end, bracketed)
	}
	mode.Write([]byte("\x1b[?2004h"))
	body, end, bracketed = mode.encode("a\nb")
	if string(body)+string(end) != "\x1b[200~a\nb\x1b[201~" || !bracketed {
		t.Fatalf("with paste on, text must be bracketed and say so, got %q%q bracketed=%v", body, end, bracketed)
	}
}

func TestTheModeIsReadFromCommandsAndNotFromWhatTheChildPrints(t *testing.T) {
	// A string sequence - a window title, a tmux passthrough - carries text the terminal
	// shows or forwards, not commands it obeys. Matching the mode bytes inside one lets a
	// title turn bracketing off on a session that still has it on, and a passthrough turn
	// it on where nothing did - after which an injection puts a literal ESC[200~ into the
	// input box.
	for _, c := range []struct {
		name   string
		writes []string
		on     bool
	}{
		{"the child turns it on", []string{"\x1b[?2004h"}, true},
		{"a title that happens to contain the off sequence", []string{"\x1b[?2004h", "\x1b]0;claude \x1b[?2004l here\x07"}, true},
		{"a tmux passthrough that contains the on sequence", []string{"\x1bPtmux;\x1b\x1b[?2004h\x1b\\"}, false},
		{"a title split across two writes", []string{"\x1b[?2004h", "\x1b]0;claude \x1b[?20", "04l here\x07"}, true},
		{"the child really does turn it off after a title", []string{"\x1b[?2004h", "\x1b]0;claude\x07", "\x1b[?2004l"}, false},
		// An opener with no terminator, printed as part of something rendered. A payload
		// is printable, so the line break after it says it was never a string, and the
		// mode changes that follow are still commands.
		{"a stray opener does not hide the mode for good", []string{"output \x1b] more\r\n", "\x1b[?2004h"}, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			mode := newPasteMode()
			for _, w := range c.writes {
				mode.Write([]byte(w))
			}
			if mode.enabled() != c.on {
				t.Fatalf("after %q: enabled=%v, want %v", c.writes, mode.enabled(), c.on)
			}
		})
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
		code, err := wrapped.run(terminalSlave, stdout, make(chan os.Signal))
		if err != nil {
			t.Errorf("run: %v", err)
		}
		exited <- code
	}()

	// Every request names the process it is for. A body that names none is given this
	// child's, so each test says only what it is about; the ones about naming the wrong
	// process say so themselves.
	ask := func(body string) response {
		t.Helper()
		var fields map[string]any
		if json.Unmarshal([]byte(body), &fields) == nil {
			if _, named := fields["pid"]; !named {
				fields["pid"] = wrapped.cmd.Process.Pid
				encoded, err := json.Marshal(fields)
				if err != nil {
					t.Fatalf("cannot encode %v: %v", fields, err)
				}
				body = string(encoded)
			}
		}
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

// recorded runs the test binary itself as the wrapped child, in the part of it that is
// TestHelperRecorder, and hands back what the child read, one entry per read.
//
// The child is Claude Code's shape as far as input goes: its terminal is raw and it has
// asked for bracketed paste. It pauses between reads, so writes fritter did not wait on
// pile up in the queue and come out of the next read together - which is how a real child
// a moment slow reads them, and what fritter has to prevent.
func recorded(t *testing.T, reads int) (*Wrapped, func(string) response, func() []string) {
	t.Helper()
	log := filepath.Join(t.TempDir(), "reads")
	t.Setenv("FRITTER_RECORD", log)
	t.Setenv("FRITTER_RECORD_READS", strconv.Itoa(reads))
	wrapped, ask, exited := wrap(t, os.Args[0], "-test.run=^TestHelperRecorder$")
	defer func() {
		if t.Failed() {
			_ = wrapped.cmd.Process.Kill()
		}
	}()
	// The child says it is ready by turning bracketing on, which it does once its
	// terminal is raw; until then a Ctrl-S would be flow control and not a key.
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		if _, _, bracketed := wrapped.paste.encode(""); bracketed {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the recording child never turned bracketed paste on")
		}
	}
	read := func() []string {
		t.Helper()
		select {
		case code := <-exited:
			if code != 3 {
				t.Fatalf("the recording child exited %d, not the 3 it exits with after %d reads", code, reads)
			}
		case <-time.After(10 * time.Second):
			_ = wrapped.cmd.Process.Kill()
			t.Fatalf("the recording child did not make %d reads", reads)
		}
		body, err := os.ReadFile(log)
		if err != nil {
			t.Fatalf("cannot read what the child read: %v", err)
		}
		var got []string
		for _, line := range strings.Split(strings.TrimSuffix(string(body), "\n"), "\n") {
			chunk, err := strconv.Unquote(line)
			if err != nil {
				t.Fatalf("cannot read the child's record %q: %v", line, err)
			}
			got = append(got, chunk)
		}
		return got
	}
	return wrapped, ask, read
}

func TestHelperRecorder(t *testing.T) {
	log := os.Getenv("FRITTER_RECORD")
	if log == "" {
		t.Skip("not a test: run as the wrapped child by recorded")
	}
	if _, err := term.MakeRaw(0); err != nil {
		os.Exit(20)
	}
	if _, err := os.Stdout.Write(pasteOn); err != nil {
		os.Exit(21)
	}
	reads, _ := strconv.Atoi(os.Getenv("FRITTER_RECORD_READS"))
	out, err := os.Create(log)
	if err != nil {
		os.Exit(22)
	}
	buf := make([]byte, 64*1024)
	for i := 0; i < reads; i++ {
		time.Sleep(50 * time.Millisecond)
		n, err := os.Stdin.Read(buf)
		if err != nil {
			os.Exit(23)
		}
		fmt.Fprintf(out, "%q\n", buf[:n])
	}
	out.Close()
	// A session does not end the instant it reads its last key, and fritter looks at the
	// input queue to see that it was read: a terminal whose session has ended cannot be
	// looked at, and what it last held is then not known to have been read.
	time.Sleep(300 * time.Millisecond)
	os.Exit(3)
}

func TestTextGoesIntoAnEmptiedBoxOneStepPerRead(t *testing.T) {
	// Each step is read on its own. Read together, Claude Code took `a`, Ctrl-S and the
	// paste for one paste: the Ctrl-S was stripped, the `a` stayed at the front of the
	// message, and the Enter was held for review.
	//
	// Except the Enter, which is read with the marker that closes the paste: read apart from
	// it, the session took the Enter before the paste was in the box and sent nothing.
	_, ask, reads := recorded(t, 6)
	if answer := ask(`{"kind":"text","text":"two\nlines","submit":true}`); !answer.OK {
		t.Fatalf("the text was refused: %s", answer.Reason)
	}
	want := []string{"a\x13", "a\x13", "\x13", "\x7f", "\x1b[200~two\nlines", "\x1b[201~\r"}
	if got := reads(); !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestTextLeftUnsentIsClosedAndNotSent(t *testing.T) {
	// Unsent, the paste is closed and nothing presses Enter.
	_, ask, reads := recorded(t, 6)
	if answer := ask(`{"kind":"text","text":"draft","submit":false}`); !answer.OK {
		t.Fatalf("the text was refused: %s", answer.Reason)
	}
	want := []string{"a\x13", "a\x13", "\x13", "\x7f", "\x1b[200~draft", "\x1b[201~"}
	if got := reads(); !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestWhatTheUserTypedIsReadApartFromWhatFollowsIt(t *testing.T) {
	// Keys the user typed just before a request can still be waiting when it takes the
	// lock. Read together with the Ctrl-S after them, they are one read the child may take
	// for a paste.
	wrapped, ask, reads := recorded(t, 7)
	if _, err := wrapped.Write([]byte("half a thought")); err != nil {
		t.Fatalf("cannot type as the user: %v", err)
	}
	if answer := ask(`{"kind":"text","text":"mine","submit":false}`); !answer.OK {
		t.Fatalf("the text was refused: %s", answer.Reason)
	}
	want := []string{"half a thought", "a\x13", "a\x13", "\x13", "\x7f", "\x1b[200~mine", "\x1b[201~"}
	if got := reads(); !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestTheExitCodeIsTheChildsOwn(t *testing.T) {
	// The child reads one line and exits with 3, so the test proves both that the text
	// arrived and that fritter did not invent an exit code of its own.
	_, ask, exited := wrap(t, "sh", "-c", "read line; sleep 0.3; exit 3")
	if answer := ask(`{"kind":"key","key":"enter"}`); !answer.OK {
		t.Fatalf("the key was refused: %s", answer.Reason)
	}
	select {
	case code := <-exited:
		if code != 3 {
			t.Fatalf("exit code %d, want the child's own 3", code)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited; the key probably never arrived")
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

// unhurried is a stdout that takes its time, which is the case the drain exists for: the
// child is reaped the moment it exits, and whether what it printed has finished being
// written out is a separate question with no ordering between them.
type unhurried struct {
	mu  sync.Mutex
	got []byte
}

func (u *unhurried) Write(p []byte) (int, error) {
	time.Sleep(100 * time.Millisecond)
	u.mu.Lock()
	defer u.mu.Unlock()
	u.got = append(u.got, p...)
	return len(p), nil
}

func (u *unhurried) written() string {
	u.mu.Lock()
	defer u.mu.Unlock()
	return string(u.got)
}

func TestRunReturnsOnlyOnceTheChildsOutputHasBeenWritten(t *testing.T) {
	// cmd.Wait comes back when the child is reaped, which says nothing about the copy
	// still in flight behind it. Returning there hands the caller a finished session
	// whose last words have not been written yet - and the caller's next move is to exit.
	slow := &unhurried{}
	_, _, exited := wrapOnto(t, slow, "sh", "-c", "echo done")
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited")
	}
	if !strings.Contains(slow.written(), "done") {
		t.Fatalf("run returned with the output still going out; stdout had %q", slow.written())
	}
}

// TestHelperFritter is not a test. It is fritter's own main, run as a subprocess by the
// test below, because the two things that test asserts - that the last of the child's
// output is written, and that the socket goes with the process - are properties of the
// program ending, and nothing that keeps running can demonstrate them.
func TestHelperFritter(t *testing.T) {
	if os.Getenv("FRITTER_HELPER") != "1" {
		t.Skip("not a test: run as a subprocess by TestTheProgramPrintsTheLastWordAndTakesItsSocketWithIt")
	}
	os.Exit(run(strings.Split(os.Getenv("FRITTER_HELPER_ARGS"), "\x1f"), os.Stdin, os.Stdout))
}

func TestTheProgramPrintsTheLastWordAndTakesItsSocketWithIt(t *testing.T) {
	// cmd.Wait returns when the child is reaped, which is before the last of what it
	// printed has come back through the pty, so without the drain the shortest possible
	// session - print one word and quit - prints nothing at all. The process has to
	// actually exit for that to show: a harness that keeps running keeps the copier
	// running too, and the copier finishes either way.
	dir := shortTempDir(t)
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--\x1fsh\x1f-c\x1fecho done; exit 3",
	)
	terminal, err := pty.Start(fritter)
	if err != nil {
		t.Fatalf("cannot start fritter on a terminal: %v", err)
	}
	defer terminal.Close()

	printed, _ := io.ReadAll(terminal)
	err = fritter.Wait()
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		t.Fatalf("fritter should have carried its child's exit code out; got %v", err)
	}
	if code := exitCode(exit); code != 3 {
		t.Errorf("exit code %d, want the child's own 3", code)
	}
	if !strings.Contains(string(printed), "done") {
		t.Errorf("the child printed \"done\" and the terminal saw %q", printed)
	}
	// A socket left behind outlives the session it addressed, and the next caller to dial
	// it reaches nothing while believing it reached a session.
	left, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	if len(left) != 0 {
		t.Errorf("fritter left %d entries in %s behind it", len(left), dir)
	}
}

func TestMultiLineTextIsRefusedWhenTheSessionWillNotBracketIt(t *testing.T) {
	// A shell never turns bracketed paste on, so a newline here is an Enter. Answering ok
	// would tell hands one message was sent where several separate prompts were.
	wrapped, ask, exited := wrap(t, "sh", "-c", "sleep 30")
	defer ending(t, wrapped, exited)

	answer := ask(`{"kind":"text","text":"first\nsecond","submit":true}`)
	if answer.OK {
		t.Fatal("a multi-line draft was accepted into a session that cannot take one whole")
	}
	if !strings.Contains(answer.Reason, "bracketed paste") {
		t.Fatalf("the refusal must say why, got %q", answer.Reason)
	}
	nothingTyped(t, answer)
}

// ending kills a child that reads nothing on purpose, which nothing else would ever end,
// and waits for run to let go of the test's terminal.
func ending(t *testing.T, wrapped *Wrapped, exited chan int) {
	t.Helper()
	_ = wrapped.cmd.Process.Kill()
	select {
	case <-exited:
	case <-time.After(10 * time.Second):
		t.Error("the child outlived being killed")
	}
}

// onlyEnterArrived presses Enter and checks it is the one thing the child ever read, which
// is how a test shows that every request before it was turned away untyped.
func onlyEnterArrived(t *testing.T, ask func(string) response, reads func() []string) {
	t.Helper()
	if answer := ask(`{"kind":"key","key":"enter"}`); !answer.OK {
		t.Fatalf("the Enter was refused: %s", answer.Reason)
	}
	if got := reads(); !slices.Equal(got, []string{"\r"}) {
		t.Fatalf("the child read %q; something refused above reached it", got)
	}
}

// nothingTyped checks a refusal says, as a value and not only in its reason, that none of
// the request reached the session: the caller resends on that alone.
func nothingTyped(t *testing.T, answer response) {
	t.Helper()
	if answer.Typed != typedNothing {
		t.Fatalf("a refusal that typed nothing says typed %q", answer.Typed)
	}
}

func TestAFailedWriteSaysWhetherTheRequestMayBeInTheSession(t *testing.T) {
	// A caller resends a refused request only when this says nothing of it was typed, so
	// it must say maybe wherever the request's own keys, or the text before them, may be
	// in the box - and nothing only where they cannot be.
	steps := textSteps([]byte("hi"), []byte{}, true)
	emptied, body, closing := steps[len(emptying)-1], steps[len(emptying)], steps[len(emptying)+1]
	key := keyStep(keystrokes["enter"])
	none := delivery{how: partway, landed: 0, of: 2, why: errors.New("x")}
	some := delivery{how: partway, landed: 1, of: 2, why: errors.New("x")}
	unknown := delivery{how: unknowable, of: 2, why: errors.New("x")}
	for _, c := range []struct {
		name string
		at   step
		how  delivery
		want typed
	}{
		{"emptying, even with its keys in the box", emptied, unknown, typedNothing},
		{"the text, none of it written", body, none, typedNothing},
		{"the text, part of it written", body, some, typedMaybe},
		{"the text, not known how much", body, unknown, typedMaybe},
		{"the Enter after the text, none of it written", closing, none, typedMaybe},
		{"a key, none of it written", key, none, typedNothing},
		{"a key, not known whether it was", key, unknown, typedMaybe},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := c.at.left(c.how); got != c.want {
				t.Fatalf("left %q, want %q", got, c.want)
			}
		})
	}
}

func TestTextThatIsNotCharactersIsRefusedRatherThanTyped(t *testing.T) {
	// text is characters and newlines. A control byte in it is a keystroke in text's
	// clothes: an ESC ends the bracketing early, so everything after it is typed and
	// submitted on its own, and a 0x03 is a Ctrl-C. Answered ok, one message would arrive
	// as several, or as something nobody asked to send.
	_, ask, reads := recorded(t, 1)

	for _, c := range []struct{ name, body string }{
		{"the marker that ends a paste", `{"kind":"text","text":"look at \u001b[201~ this","submit":true}`},
		{"an interrupt", `{"kind":"text","text":"a\u0003b","submit":true}`},
		{"a carriage return, which submits", `{"kind":"text","text":"first\rsecond","submit":true}`},
		{"a tab, which is a key", `{"kind":"text","text":"a\tb","submit":true}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			answer := ask(c.body)
			if answer.OK {
				t.Fatal("this was accepted as text and sent to the session")
			}
			if !strings.Contains(answer.Reason, "control byte") {
				t.Fatalf("the reason must say what it found, got %q", answer.Reason)
			}
			nothingTyped(t, answer)
		})
	}
	onlyEnterArrived(t, ask, reads)
}

func TestASessionThatIsNotReadingItsInputIsSaidSoRatherThanWaitedOn(t *testing.T) {
	// Every write waits for the child to read it, which a stopped or wedged session never
	// does. Waiting there with no bound hangs the request and everything behind it: the next
	// caller in is hands sending the ctrl_c that was supposed to be the way back.
	//
	// Raw, because a cooked terminal counts nothing until a line is in and the wait would
	// be no wait; Claude Code runs raw.
	wrapped, ask, exited := wrap(t, "sh", "-c", "sleep 30")
	if _, err := term.MakeRaw(int(wrapped.master.Fd())); err != nil {
		t.Fatalf("cannot put the session's terminal into raw mode: %v", err)
	}
	defer ending(t, wrapped, exited)

	stuck := ask(`{"kind":"text","text":"hello","submit":true}`)
	if stuck.OK {
		t.Fatal("a session that reads nothing was reported as taking the text")
	}
	if !strings.Contains(stuck.Reason, "not reading its input") {
		t.Fatalf("the reason must say the session is not reading, got %q", stuck.Reason)
	}
	// The first emptying step is in the queue, unread. The caller has to hear that the box
	// was being changed under the user, and that none of its text is there.
	if !strings.Contains(stuck.Reason, "none of the text was typed") || !strings.Contains(stuck.Reason, "may be gone") {
		t.Fatalf("the reason must say the box was being emptied and the text not typed, got %q", stuck.Reason)
	}
	nothingTyped(t, stuck)

	// The write is still out there and cannot be taken back, so the next request is
	// refused at once rather than queued behind it - queued it would wait just as long,
	// and two half-written messages interleave into one nobody can attribute.
	started := time.Now()
	behind := ask(`{"kind":"key","key":"ctrl_u"}`)
	if behind.OK {
		t.Fatal("a chord was reported as delivered while a write was still stuck")
	}
	if waited := time.Since(started); waited > typeGrace {
		t.Fatalf("the next request waited %s behind the stuck one", waited)
	}
	if !strings.Contains(behind.Reason, "has not finished") {
		t.Fatalf("the reason must say what it is behind, got %q", behind.Reason)
	}
	nothingTyped(t, behind)
}

func TestRequestsThatNameNothingRealAreRefusedWithAReason(t *testing.T) {
	_, ask, reads := recorded(t, 1)

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
			nothingTyped(t, answer)
		})
	}
	onlyEnterArrived(t, ask, reads)
}

func TestARequestForAnotherProcessIsNotTypedIntoThisOne(t *testing.T) {
	// The address is inherited, and so is carried by any session started from inside
	// this one. A request naming that other session's process must not land here.
	wrapped, ask, reads := recorded(t, 1)

	for _, c := range []struct{ name, body string }{
		{"another process", fmt.Sprintf(`{"pid":%d,"kind":"text","text":"intruder","submit":true}`, wrapped.cmd.Process.Pid+1)},
		{"no process at all", `{"pid":0,"kind":"key","key":"enter"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			answer := ask(c.body)
			if answer.OK {
				t.Fatal("a request for another process was typed into this one")
			}
			if !strings.Contains(answer.Reason, "nothing was typed") {
				t.Fatalf("the refusal must say nothing was typed, got %q", answer.Reason)
			}
			nothingTyped(t, answer)
		})
	}
	onlyEnterArrived(t, ask, reads)
}

func TestASubmitTheSessionWouldNotSendIsRefusedBeforeAnythingIsTyped(t *testing.T) {
	// A Return after a backslash is a newline, and one under a completion list picks an
	// entry; either way the text would sit in the box under an ok that said it was sent.
	_, ask, reads := recorded(t, 1)

	for _, text := range []string{`continue me \`, "look at @src/ha", "fix @", "see #12", "run :ab"} {
		t.Run(text, func(t *testing.T) {
			encoded, _ := json.Marshal(map[string]any{"kind": "text", "text": text, "submit": true})
			answer := ask(string(encoded))
			if answer.OK {
				t.Fatalf("%q was reported sent", text)
			}
			if !strings.Contains(answer.Reason, "nothing was typed") {
				t.Fatalf("the refusal must say nothing was typed, got %q", answer.Reason)
			}
			nothingTyped(t, answer)
		})
	}
	onlyEnterArrived(t, ask, reads)
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

func TestClosingTheControlSocketTakesItsDirectoryToo(t *testing.T) {
	// That the program does this on its way out is the test above; this is the piece it
	// calls, which must leave nothing behind for the next caller to dial into.
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

func TestASessionThatCannotBeAttachedIsNotLeftRunning(t *testing.T) {
	// The child starts before the terminal is put into raw mode, so a stdin that is not a
	// terminal fails after there is already a session to end.
	wrapped, err := start([]string{"sh", "-c", "sleep 30"}, nil)
	if err != nil {
		t.Fatalf("cannot start: %v", err)
	}
	notATerminal, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatalf("cannot make a stdin: %v", err)
	}
	defer notATerminal.Close()
	if _, err := wrapped.run(notATerminal, io.Discard, make(chan os.Signal)); err == nil {
		t.Fatal("run attached to a file as though it were a terminal")
	}
	if wrapped.cmd.ProcessState == nil {
		_ = wrapped.cmd.Process.Kill()
		t.Fatal("run returned with the session it started still running")
	}
}
