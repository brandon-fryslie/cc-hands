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
	pasted, bracketed := mode.encode("a\nb")
	if string(pasted) != "a\nb" || bracketed {
		t.Fatalf("with paste off, text must go as it is and say so, got %q bracketed=%v", pasted, bracketed)
	}
	mode.Write([]byte("\x1b[?2004h"))
	pasted, bracketed = mode.encode("a\nb")
	if string(pasted) != "\x1b[200~a\nb\x1b[201~" || !bracketed {
		t.Fatalf("with paste on, text must be bracketed and say so, got %q bracketed=%v", pasted, bracketed)
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
// asked for bracketing.
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
	// terminal is raw.
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		if _, bracketed := wrapped.paste.encode(""); bracketed {
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
	os.Exit(3)
}

func TestTextIsPastedAndSentAsOneWrite(t *testing.T) {
	// The text and the Return behind it, as someone at the keyboard would type them. The
	// child reads them together, so the Return sends what the paste put in the box.
	_, ask, reads := recorded(t, 1)
	if answer := ask(`{"kind":"text","text":"two\nlines"}`); !answer.OK {
		t.Fatalf("the text was refused: %s", answer.Reason)
	}
	want := []string{"\x1b[200~two\nlines\x1b[201~\r"}
	if got := reads(); !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestAnEscapeIsReadAloneWhateverIsTypedStraightAfterIt(t *testing.T) {
	// An ESC read together with the byte behind it is a chord, not an Escape: Claude Code
	// took Escape then Ctrl-C, sent at once, as Alt+Ctrl-C and stopped nothing. The child
	// reads every 50ms, so keys sent within that time of each other arrive in one read
	// unless fritter keeps them apart.
	_, ask, reads := recorded(t, 2)
	for _, key := range []string{"escape", "ctrl_c"} {
		if answer := ask(`{"kind":"key","key":"` + key + `"}`); !answer.OK {
			t.Fatalf("the key %s was refused: %s", key, answer.Reason)
		}
	}
	if got, want := reads(), []string{"\x1b", "\x03"}; !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestACommandIsTypedAndOnlyItsTextPasted(t *testing.T) {
	// A long paste is folded into a placeholder that hides whatever it began with, so the
	// command goes as keys and only what follows it as a paste.
	_, ask, reads := recorded(t, 1)
	if answer := ask(`{"kind":"command","command":"/btw","text":"two\nlines"}`); !answer.OK {
		t.Fatalf("the command was refused: %s", answer.Reason)
	}
	want := []string{"/btw \x1b[200~two\nlines\x1b[201~\r"}
	if got := reads(); !slices.Equal(got, want) {
		t.Fatalf("the child read %q, want %q", got, want)
	}
}

func TestACommandWithNoTextIsTypedAlone(t *testing.T) {
	_, ask, reads := recorded(t, 1)
	if answer := ask(`{"kind":"command","command":"/clear"}`); !answer.OK {
		t.Fatalf("the command was refused: %s", answer.Reason)
	}
	if got, want := reads(), []string{"/clear\r"}; !slices.Equal(got, want) {
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

// flaky is a terminal that fails the writes it is told to, and takes the rest.
type flaky struct {
	failures []error
	got      string
}

func (f *flaky) Write(p []byte) (int, error) {
	if len(f.failures) > 0 {
		err := f.failures[0]
		f.failures = f.failures[1:]
		return 0, err
	}
	f.got += string(p)
	return len(p), nil
}

func TestOnlyAHungUpTerminalStopsBeingWrittenTo(t *testing.T) {
	// A write that fails in passing - a descriptor another process on the same terminal
	// made non-blocking answers EAGAIN - must not blank the user's window for the rest of
	// the session while the child carries on underneath it.
	passing := &flaky{failures: []error{syscall.EAGAIN}}
	shown := &screen{out: passing}
	shown.Write([]byte("lost "))
	shown.Write([]byte("shown"))
	if passing.got != "shown" {
		t.Errorf("after a passing failure the terminal was shown %q, want %q", passing.got, "shown")
	}

	// A hung-up terminal fails every write from then on, and fritter's SIGHUP is already
	// ending the session; what the child still prints has nowhere to go but away.
	gone := &flaky{failures: []error{syscall.EIO}}
	shown = &screen{out: gone}
	shown.Write([]byte("lost "))
	shown.Write([]byte("also lost"))
	if gone.got != "" {
		t.Errorf("a hung-up terminal was written to again: %q", gone.got)
	}
}

func TestClosingTheTerminalEndsTheSessionAndTakesItsSocketWithIt(t *testing.T) {
	// The child is Claude Code's shape on its way out: told its terminal hung up, it has
	// more to write than a pty holds before it can exit. With the terminal gone, fritter
	// is the only reader that output has; if fritter stops reading, the child never exits
	// and neither does fritter, and both are left running with nobody able to see them.
	dir := shortTempDir(t)
	fritter := exec.Command(os.Args[0], "-test.run=TestHelperFritter")
	fritter.Env = append(os.Environ(),
		"FRITTER_HELPER=1",
		"FRITTER_HELPER_ARGS=--socket-dir\x1f"+dir+"\x1f--\x1fsh\x1f-c\x1f"+
			`trap 'head -c 1048576 /dev/zero; exit 0' HUP; echo "ready $$"; read line`,
	)
	terminal, err := pty.Start(fritter)
	if err != nil {
		t.Fatalf("cannot start fritter on a terminal: %v", err)
	}
	defer fritter.Process.Kill()
	exited := make(chan error, 1)
	go func() { exited <- fritter.Wait() }()
	// One bound over the whole session. A fritter that never prints, or never exits, is
	// killed, which ends the read or the wait below and fails the test rather than hanging it.
	watchdog := time.AfterFunc(10*time.Second, func() { _ = fritter.Process.Kill() })
	defer watchdog.Stop()

	var printed string
	for at := -1; at < 0 || !strings.Contains(printed[at:], "\n"); at = strings.Index(printed, "ready") {
		chunk := make([]byte, 256)
		n, err := terminal.Read(chunk)
		if err != nil {
			t.Fatalf("the child never said it was ready; the terminal saw %q: %v", printed, err)
		}
		printed += string(chunk[:n])
	}
	var child int
	if _, err := fmt.Sscanf(printed[strings.Index(printed, "ready"):], "ready %d", &child); err != nil {
		t.Fatalf("cannot read the child's pid from %q: %v", printed, err)
	}
	// The child leads a process group of its own, which holds whatever it started too.
	defer syscall.Kill(-child, syscall.SIGKILL)

	// What a terminal emulator does when its window closes, and tmux when its pane is
	// killed: the terminal goes, and fritter, whose controlling terminal it was, is hung up.
	terminal.Close()

	<-exited
	if !watchdog.Stop() {
		t.Fatal("fritter and its child outlived the terminal they ran in")
	}
	if err := syscall.Kill(child, 0); !errors.Is(err, syscall.ESRCH) {
		t.Errorf("fritter exited and left its child %d running: %v", child, err)
	}
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

	answer := ask(`{"kind":"text","text":"first\nsecond"}`)
	if answer.OK {
		t.Fatal("a multi-line draft was accepted into a session that cannot take one whole")
	}
	if !strings.Contains(answer.Reason, "bracketed paste") {
		t.Fatalf("the refusal must say why, got %q", answer.Reason)
	}
}

func TestPastingSaysWhetherTheChildTurnedBracketedPasteOnAndTypesNothing(t *testing.T) {
	// hands types a brain's first prompt once the brain has turned bracketing on, which Claude
	// Code does once its input is up: before, the prompt would go unbracketed into an input
	// that is not there yet.
	wrapped, ask, exited := wrap(t, "sh", "-c", "sleep 30")
	defer ending(t, wrapped, exited)
	if answer := ask(`{"kind":"pasting"}`); answer.OK || !strings.Contains(answer.Reason, "bracketed paste") {
		t.Fatalf("a shell, which never turns bracketing on, was said to have it: %+v", answer)
	}

	_, askRecorded, reads := recorded(t, 1)
	if answer := askRecorded(`{"kind":"pasting"}`); !answer.OK {
		t.Fatalf("a child that turned bracketing on was said not to have: %s", answer.Reason)
	}
	onlyEnterArrived(t, askRecorded, reads)
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
func TestTextThatIsNotCharactersIsRefusedRatherThanTyped(t *testing.T) {
	// text is characters and newlines. A control byte in it is a keystroke in text's
	// clothes: an ESC ends the bracketing early, so everything after it is typed and
	// submitted on its own, and a 0x03 is a Ctrl-C. Answered ok, one message would arrive
	// as several, or as something nobody asked to send.
	_, ask, reads := recorded(t, 1)

	for _, c := range []struct{ name, body string }{
		{"the marker that ends a paste", `{"kind":"text","text":"look at \u001b[201~ this"}`},
		{"an interrupt", `{"kind":"text","text":"a\u0003b"}`},
		{"a carriage return, which submits", `{"kind":"text","text":"first\rsecond"}`},
		{"a tab, which is a key", `{"kind":"text","text":"a\tb"}`},
		{"a command's text", `{"kind":"command","command":"/btw","text":"a\u0003b"}`},
		{"a command", `{"kind":"command","command":"/b\u001btw","text":"a"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			answer := ask(c.body)
			if answer.OK {
				t.Fatal("this was accepted as text and sent to the session")
			}
			if !strings.Contains(answer.Reason, "control byte") {
				t.Fatalf("the reason must say what it found, got %q", answer.Reason)
			}
		})
	}
	onlyEnterArrived(t, ask, reads)
}

func TestRequestsThatNameNothingRealAreRefusedWithAReason(t *testing.T) {
	_, ask, reads := recorded(t, 1)

	for _, c := range []struct{ name, body, says string }{
		{"an unknown key", `{"kind":"key","key":"f13"}`, "no key named"},
		{"an unknown kind", `{"kind":"shout","text":"hi"}`, "no request kind named"},
		{"not json at all", `nonsense`, "cannot read the request"},
		{"a command of no word", `{"kind":"command","text":"hi"}`, "one word"},
		{"a command of two words", `{"kind":"command","command":"/btw hi","text":"there"}`, "one word"},
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
	onlyEnterArrived(t, ask, reads)
}

func TestARequestForAnotherProcessIsNotTypedIntoThisOne(t *testing.T) {
	// The address is inherited, and so is carried by any session started from inside
	// this one. A request naming that other session's process must not land here.
	wrapped, ask, reads := recorded(t, 1)

	for _, c := range []struct{ name, body string }{
		{"another process", fmt.Sprintf(`{"pid":%d,"kind":"text","text":"intruder"}`, wrapped.cmd.Process.Pid+1)},
		{"no process at all", `{"pid":0,"kind":"key","key":"enter"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			answer := ask(c.body)
			if answer.OK {
				t.Fatal("a request for another process was typed into this one")
			}
			if !strings.Contains(answer.Reason, "types into process") {
				t.Fatalf("the refusal must name the process it types into, got %q", answer.Reason)
			}
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
