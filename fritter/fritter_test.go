package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
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
		// Ctrl-U kills back to the start of the line the cursor is on, and that is the
		// displayed line: measured, 250 characters in a 100-column terminal lost one row
		// to a single press and kept 192. Where the cursor is and how wide the terminal
		// is are both invisible here, so no press of it proves the box is empty.
		{"ctrl-u cannot prove the box is empty", []string{"oops", "\x15"}, false},
		{"not on a box with more than one line in it", []string{"aaa", "\n", "bbb", "\x15"}, false},
		{"and not on one that was already past accounting for", []string{"\x1b[A", "\x15"}, false},
		{"ctrl-c does, however many lines are in it", []string{"aaa", "\n", "bbb", "\x03"}, true},
		{"escape leaves the box alone", []string{"oops", "\x1b"}, false},
		{"a word killed leaves the count standing", []string{"alpha beta", "\x17"}, false},
		// A bare newline is Ctrl-J, which puts one in the box rather than sending it.
		{"ctrl-j is a newline in the box, not a submit", []string{"hello", "\n"}, false},
		{"ctrl-j then a real Return sends it", []string{"hello", "\n", "world", "\r"}, true},
		{"ctrl-j into an empty box puts a character in it", []string{"\n"}, false},
		{"ctrl-j and a Return arriving in one read", []string{"hello\nworld\r"}, true},
		{"an empty read changes nothing", []string{"half", ""}, false},
		// Backspace looks like the one key whose effect is a number - one character, every
		// time - and it is not. The child reads it as
		// `backspace(){if(this.isAtStart())return this;...}`, so a press with the cursor at
		// the start of the box takes nothing out of it, and where the cursor is is not
		// something stdin says. Counted, these free a line with the user's words still in
		// it, which is the one mistake with no recovery.
		{"backspaced back to what might be empty", []string{"abc", "\x7f\x7f\x7f"}, false},
		{"backspaced most of the way", []string{"abc", "\x7f\x7f"}, false},
		{"backspaced past where empty would have been", []string{"a", "\x7f\x7f\x7f"}, false},
		{"taken back from the start of the box, where it takes nothing",
			[]string{"x", "\x01", "\x7f"}, false},
		// Holding the key down past the start of a line is an autorepeat, not a corner:
		// six of these erase `hello `, and the other five land at offset 0 and do nothing.
		{"held down past the start of a line", []string{
			"hello world", "\x1b[D\x1b[D\x1b[D\x1b[D\x1b[D", strings.Repeat("\x7f", 11)}, false},
		{"and a Return after all of it still hands the line back",
			[]string{"abc", "\x7f\x7f\x7f", "\r"}, true},

		{"the window losing focus", []string{"\x1b[O"}, true},
		{"the window gaining focus", []string{"\x1b[I"}, true},
		{"focus lost while a line is held", []string{"half", "\x1b[O"}, false},
		{"an sgr mouse report", []string{"\x1b[<35;89;12M"}, true},
		{"an x10 mouse report whose bytes are printable", []string{"\x1b[M !!"}, true},
		{"a cursor position report", []string{"\x1b[24;80R"}, true},
		{"a device attributes report", []string{"\x1b[?1;2c"}, true},
		{"a colour query answered", []string{"\x1b]11;rgb:1b1b/1b1b/1b1b\x07"}, true},
		{"a report split across two reads", []string{"\x1b[<35;8", "9;12M"}, true},
		// An ESC alone in a read is the Escape key, so `[A` after it is two characters -
		// even when it really was an arrow key whose sequence the read cut in half. That
		// holds a line that is empty, which an Enter or a ctrl_c clears. The other way
		// round frees a line that is not empty, and nothing clears that.
		{"an escape split from what might be its sequence", []string{"\x1b", "[A"}, false},
		// Up and Down are the history keys: they pull a whole previous prompt into the box,
		// so an empty box fills without a character being typed. Left and Right only move.
		{"the history keys fill the box", []string{"\x1b[A"}, false},
		{"down fills it too", []string{"\x1b[B"}, false},
		{"and so does up in application mode", []string{"\x1bOA"}, false},
		{"sideways is only the cursor", []string{"\x1b[C", "\x1b[D", "\x1bOC"}, true},
		{"home and end are only the cursor", []string{"\x1b[H", "\x1b[F"}, true},
		// Everything that is not a character and is not known to leave the box alone is
		// read as having changed it by some amount: Ctrl-Y pastes back what was killed,
		// Tab completes a path in, Delete takes one out.
		{"ctrl-y pastes back whatever was last killed", []string{"\x19"}, false},
		{"tab can complete a path into the box", []string{"\t"}, false},
		{"the delete key takes a character out", []string{"\x1b[3~"}, false},
		{"but ctrl-a and ctrl-e only move", []string{"\x01", "\x05"}, true},
		// The binary binds Ctrl-L to clearInput and pressing it changed nothing. Evidence
		// that disagrees with itself is not evidence that the box was left alone.
		{"ctrl-l is not known to leave the box alone", []string{"\x0c"}, false},
		{"escape alone is not a character", []string{"\x1b"}, true},

		{"a pasted line is characters in the box", []string{"\x1b[200~hello\nworld\x1b[201~"}, false},
		{"a newline inside a paste does not submit", []string{"\x1b[200~a\nb\x1b[201~"}, false},
		{"a paste spanning reads", []string{"\x1b[200~a\nb", "c\x1b[201~"}, false},
		{"a paste whose end marker straddles a read", []string{"\x1b[200~ab\x1b[20", "1~"}, false},
		{"a paste then submitted", []string{"\x1b[200~a\nb\x1b[201~", "\r"}, true},

		// An ESC is both the Escape key and the first byte of every sequence, and the
		// bytes alone do not say which. Guessing "sequence" lets a scan looking for a
		// terminator swallow whatever the user types next and report an empty box while
		// their words sit in it - the one mistake with no recovery. Guessing "Escape key"
		// counts the sequence that follows as typing and holds an empty line for good.
		// These are the cases that catch each guess.
		{"escape, then a message that opens like a string sequence", []string{"\x1b", "P", "l", "e", "a", "s", "e"}, false},
		{"escape, then a message that opens like another one", []string{"\x1b", "]drop the table"}, false},
		{"escape, then the pointer moves", []string{"\x1b", "\x1b[<0;45;12M"}, true},
		{"a box proved empty by a Return is accounted for again", []string{"\x1b[A", "\r"}, true},
		{"and one proved empty by ctrl-c is too", []string{"\x1b[A", "\x03"}, true},
		{"escape, then the window loses focus", []string{"\x1b", "\x1b[O"}, true},
		{"alt-up arrives as two escapes and a history key", []string{"\x1b\x1b[A"}, false},
		{"typed, escape, then submitted", []string{"hello", "\x1b", "\r"}, true},
		{"typed, escape, then cleared", []string{"hello", "\x1b", "\x03"}, true},
		{"typed, escape, then backspaced back to what might be empty",
			[]string{"ab", "\x1b", "\x7f\x7f"}, false},
		{"an answer longer than any key holds the line rather than freeing it", []string{"\x1b]" + strings.Repeat("A", 4095), "BBB\x07"}, false},
		{"a terminal answer split across reads is still not typing", []string{"\x1b]11;rgb:1b1b/", "1b1b/1b1b\x07"}, true},
		{"a terminal answer split at its terminator is still not typing", []string{"\x1b]11;rgb:1b1b/1b1b/1b1b\x1b", "\\"}, true},

		// An ESC that arrived alone was the Escape key, so the next read is a new keypress
		// rather than the rest of a chord. Fusing them swallows the character silently: the
		// count stays at zero and the line reads free while the user's word sits in the box.
		{"escape, then a character typed", []string{"\x1b", "a"}, false},
		{"escape, then two typed and one taken back", []string{"\x1b", "ab", "\x7f"}, false},
		// Alt and a letter, and the Escape key followed by that letter, are the same two
		// bytes in the same read - ssh and tmux deliver everything typed within a round
		// trip together. Read as a chord, the letter is lost off the count.
		{"alt and a letter arriving together is read as the letter", []string{"\x1ba"}, false},
		// Except when the second byte is a control byte, where the chord is real and
		// Option-Enter puts a newline in the box instead of submitting it.
		{"alt and Enter arriving together do not submit", []string{"abc", "\x1b\r"}, false},
		// `O` and `[` are ordinary characters as well as the second byte of an arrow key.
		// Read as the sequence, these count nothing and free a line holding two letters.
		{"escape, then a word beginning with O", []string{"\x1b", "O", "k"}, false},
		// The same, arriving in one read - over ssh or tmux, everything typed within a
		// round trip does. It reads as an SS3, and an SS3 not known to be harmless holds.
		{"escape and a word beginning with O arriving together", []string{"\x1bOk"}, false},
		{"escape and a bracketed word arriving together", []string{"\x1b[x"}, false},
		// A terminal can send any key as a CSI sequence. The kitty keyboard protocol sends
		// Ctrl-Y, which pastes back what was last killed, as `ESC [ 121 ; 5 u`.
		{"a key sent as a CSI sequence is not known to be harmless", []string{"\x1b[121;5u"}, false},
		{"but the keyboard flags the terminal reports are only an answer", []string{"\x1b[?1u"}, true},
		{"and shift-tab only cycles the mode", []string{"\x1b[Z"}, true},
		{"escape, then a word beginning with a bracket", []string{"\x1b", "[", "x"}, false},
		{"escape, then a word beginning with O, backspaced to one", []string{"\x1b", "Oops", "\x7f\x7f"}, false},
		// An SS3 that ends on a control byte was never an SS3, and taking three bytes
		// regardless swallows the Enter that was the third of them.
		{"a half-read SS3 must not swallow the Enter after it", []string{"a\x1bO", "\r"}, true},

		// Claude Code 2.1.278 reads a Return after a backslash as "keep typing": the
		// backslash becomes a newline and everything already typed stays in the box.
		// Measured against the running program. Read as a submit, it empties a count that
		// is not empty and hands writes into the middle of someone's sentence.
		{"a backslash and Return continue the line", []string{"please fix the auth bug in \\", "\r"}, false},
		{"a backslash and Return arriving in one read", []string{"abc\\\r"}, false},
		{"a line continued and then really submitted", []string{"abc\\", "\r", "def", "\r"}, true},
		{"ctrl-c empties a continued line, because it empties anything", []string{"abc\\", "\r", "\x03"}, true},
		{"but ctrl-u cannot prove it emptied one", []string{"abc\\", "\r", "\x15"}, false},
		// Nothing comes off the remembered end any more, because nothing on stdin says how
		// much came off the box. So what it holds is a superset of what the line really
		// ends with - and a superset can only hold a Return that would have sent, never
		// free one that would not.
		{"a backslash taken back is remembered anyway, and holds", []string{"abc\\", "\x7f", "\r"}, false},
		{"until the Return after it", []string{"abc\\", "\x7f", "\r", "\r"}, true},
		// The child looks at the character before the cursor, and nothing here knows where
		// that is. Measured: `ab\c`, one Left, Return, and the box kept both halves.
		{"a backslash anywhere in the line's end holds the Return after it", []string{"a\\bc", "\r"}, false},
		{"and the Return after that one sends it", []string{"a\\bc", "\r", "\r"}, true},
		{"a line with no backslash in it at all is sent", []string{"abc", "\r"}, true},
		{"a pasted line ending in a backslash continues too", []string{"\x1b[200~a\\\x1b[201~", "\r"}, false},
		// And because it cannot shrink, it can no longer run past what it remembers, which
		// is the whole of what the murky flag used to cover.
		{"a Return after more was taken back than was remembered sends", []string{
			strings.Repeat("a", 20), strings.Repeat("\x7f", 18), "\r"}, true},
		{"one with a backslash still remembered does not", []string{
			"abc\\" + strings.Repeat("a", 12), strings.Repeat("\x7f", 12), "\r"}, false},

		// A Return with a completion list up does not send. The child calls
		// preventDefault() and applies the highlighted entry instead, which leaves the box
		// longer than it was and still unsent. Whether the list is up is decided by the
		// token ending at the cursor, and neither is visible here, so what is asked is
		// whether any cursor position in the remembered end would have opened one.
		{"a Return picking a file completion does not send", []string{"look at @src/ha", "\r"}, false},
		{"nor the one after it, while the token is still remembered",
			[]string{"look at @src/ha", "\r", "\r"}, false},
		{"a bare @ opens the list on every file there is", []string{"fix @", "\r"}, false},
		{"an @ the cursor could have been put back inside", []string{"@foo and more", "\r"}, false},
		{"an @ in the middle of a word opens nothing", []string{"mail bmf@example", "\r"}, true},
		{"a # with nothing after it cannot match", []string{"see #", "\r"}, true},
		{"and holds the Return once it has something", []string{"see #12", "\r"}, false},
		{"a colon in ordinary prose is not a completion", []string{"note: fix this", "\r"}, true},
		{"a colon token is", []string{"nice :smile", "\r"}, false},
		{"a slash command runs and empties the box", []string{"/compact", "\r"}, true},
		{"an ordinary prompt is still sent by its Return", []string{"what changed today", "\r"}, true},
		{"ctrl-c empties a box a completion left full", []string{"look at @src/ha", "\r", "\x03"}, true},
		{"and what it left is forgotten as the user types on", []string{
			"look at @src/ha", "\r", "and then some more words", "\r"}, true},
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
		{"ctrl-u cannot prove it clear, so it stays held", "ctrl_u", false},
		{"escape does not touch it", "escape", false},
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

func TestAKeyThatEmptiesTheBoxSettlesTheLineEvenMidSequence(t *testing.T) {
	// The line has to be recoverable from a held state, and the held state that matters is
	// the one the parser cannot resolve: an Escape, then a message that opens like a
	// terminal answer, leaves it holding bytes it cannot yet call typing. Freeing the
	// count alone is not enough - the doubt has to go with it, or the ctrl_c sent to
	// clear the line leaves the line held by the very request sent to clear it.
	line := newLineOwner()
	line.typed([]byte("\x1b"))
	line.typed([]byte("Please fix the auth bug"))
	if line.free() {
		t.Fatal("the user is typing and the parser cannot yet see it; the line is not free")
	}
	line.sent(keystrokes["ctrl_c"])
	if !line.free() {
		t.Fatal("the box was emptied and the line is still held; nothing can recover it")
	}
}

func TestAKeyDoesNotTakeTheEndOfThePasteWithIt(t *testing.T) {
	// The marker that ends a paste can be cut in half by the end of a read, and the half
	// of it that arrived is held over. Those bytes are not an unfinished sequence, so
	// emptying the box must not discard them: without the marker's first bytes the marker
	// never matches, the parser stays inside a paste that has ended, and from then on
	// every Enter counts as a pasted character instead of emptying the box. No keystroke
	// recovers from that - the session can never be typed into again.
	line := newLineOwner()
	line.typed([]byte("\x1b[200~line one"))
	line.typed([]byte(" and more\x1b[20"))
	line.sent(keystrokes["ctrl_c"])
	line.typed([]byte("1~"))
	line.typed([]byte("hi"))
	line.typed([]byte("\r"))
	if !line.free() {
		t.Fatal("the paste's end marker was discarded, so the paste never ended and Enter no longer empties the box")
	}
}

func TestAKeyDoesNotTakeThePasteItDidNotEnd(t *testing.T) {
	// Emptying the box settles what the parser was in the middle of reading, because those
	// bytes are not in the box any more. It settles nothing about a paste the user is still
	// making at their own keyboard: the terminal is going to send the rest of it either
	// way, and read outside its brackets the newlines in it are Enter presses. Each one
	// would empty a count that is not empty, and the line would read free with the tail of
	// someone's paste sitting in the box.
	line := newLineOwner()
	line.typed([]byte("\x1b[200~line one"))
	if line.free() {
		t.Fatal("a paste in progress is characters in the box")
	}
	line.sent(keystrokes["ctrl_c"])
	line.typed([]byte("\nline two\n\x1b[201~"))
	if line.free() {
		t.Fatal("the rest of the paste was read as typing and its newlines freed the line")
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

func TestEncodeBracketsOnlyWhenTheChildAcceptsItAndSaysWhichItDid(t *testing.T) {
	// The caller needs the same reading of the mode that the bytes were made with. Asked
	// twice - once to decide whether multi-line text is safe, once to encode it - the
	// child can turn bracketing off in between, and a message accepted as one paste goes
	// as several submitted prompts under an ok.
	mode := newPasteMode()
	got, bracketed := mode.encode("a\nb")
	if string(got) != "a\nb" || bracketed {
		t.Fatalf("with paste off, text must go as it is and say so, got %q bracketed=%v", got, bracketed)
	}
	mode.Write([]byte("\x1b[?2004h"))
	got, bracketed = mode.encode("a\nb")
	if string(got) != "\x1b[200~a\nb\x1b[201~" || !bracketed {
		t.Fatalf("with paste on, text must be bracketed and say so, got %q bracketed=%v", got, bracketed)
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
	if !strings.Contains(answer.Reason, "not seen to be sent") {
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

func TestTextThatIsNotCharactersIsRefusedRatherThanTyped(t *testing.T) {
	// text is characters and newlines. A control byte in it is a keystroke in text's
	// clothes: an ESC ends the bracketing early, so everything after it is typed and
	// submitted on its own, and a 0x03 is a Ctrl-C. Answered ok, one message would arrive
	// as several, or as something nobody asked to send.
	_, ask, _ := wrap(t, "sh", "-c", "IFS= read -r a; exit 0")

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
		})
	}
	if answer := ask(`{"kind":"text","text":"ordinary words","submit":true}`); !answer.OK {
		t.Fatalf("the child could not be let go: %s", answer.Reason)
	}
}

func TestASessionThatIsNotReadingItsInputIsSaidSoRatherThanWaitedOn(t *testing.T) {
	// A pty in raw mode holds a kilobyte of input, and a write that fills it blocks until
	// the child reads - which a stopped or wedged session never does. Waiting there with
	// no bound hangs the request and everything behind it: the next caller in is hands
	// sending the ctrl_c that was supposed to be the way back.
	//
	// Measured on macOS: a cooked pty takes 300 KB without blocking, a raw one blocks at
	// 1024 bytes. Claude Code runs raw, so raw is the configuration this has to hold in,
	// and the test puts the pty there rather than testing the one that cannot fail.
	wrapped, ask, exited := wrap(t, "sh", "-c", "sleep 30")
	if _, err := term.MakeRaw(int(wrapped.master.Fd())); err != nil {
		t.Fatalf("cannot put the session's terminal into raw mode: %v", err)
	}
	// The child is killed at the end whatever happens: it reads nothing on purpose, so
	// nothing else will ever end it, and run would still hold the test's terminal.
	defer func() {
		_ = wrapped.cmd.Process.Kill()
		select {
		case <-exited:
		case <-time.After(10 * time.Second):
			t.Error("the child outlived being killed")
		}
	}()

	stuck := ask(`{"kind":"text","text":"` + strings.Repeat("x", 2000) + `","submit":false}`)
	if stuck.OK {
		t.Fatal("a session that reads nothing took two kilobytes")
	}
	if !strings.Contains(stuck.Reason, "not reading its input") {
		t.Fatalf("the reason must say the session is not reading, got %q", stuck.Reason)
	}
	// The write went into a kilobyte-deep queue before it blocked, so part of the message
	// is in front of the user. "Nothing was typed" reads as "send it again", which doubles
	// the half that is already there.
	if strings.Contains(stuck.Reason, "nothing was typed") {
		t.Fatalf("a write that may have landed partly was reported as landing not at all: %q", stuck.Reason)
	}

	// The write is still out there and cannot be taken back, so the next request is
	// refused at once rather than queued behind it - queued it would wait just as long,
	// and two half-written messages interleave into one nobody can attribute.
	started := time.Now()
	behind := ask(`{"kind":"key","key":"ctrl_u"}`)
	if behind.OK {
		t.Fatal("a chord was reported as delivered while a write was still stuck")
	}
	if waited := time.Since(started); waited > writeGrace {
		t.Fatalf("the next request waited %s behind the stuck one", waited)
	}
	if !strings.Contains(behind.Reason, "has not finished") {
		t.Fatalf("the reason must say what it is behind, got %q", behind.Reason)
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

func TestARequestForAnotherProcessIsNotTypedIntoThisOne(t *testing.T) {
	// The address is inherited, and so is carried by any session started from inside
	// this one. A request naming that other session's process must not land here.
	wrapped, ask, _ := wrap(t, "sh", "-c", "IFS= read -r line; test \"$line\" = mine && exit 3 || exit 9")
	defer func() { _ = wrapped.cmd.Process.Kill() }()

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
		})
	}
	if answer := ask(`{"kind":"text","text":"mine","submit":true}`); !answer.OK {
		t.Fatalf("a request for this process was refused: %s", answer.Reason)
	}
}

func TestASubmitTheSessionWouldNotSendIsRefusedBeforeAnythingIsTyped(t *testing.T) {
	// A Return after a backslash is a newline, and one under a completion list picks an
	// entry; either way the text would sit in the box under an ok that said it was sent.
	wrapped, ask, exited := wrap(t, "sh", "-c", "IFS= read -r line; test \"$line\" = \"note: fix this\" && exit 3 || exit 9")
	defer func() { _ = wrapped.cmd.Process.Kill() }()

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
		})
	}
	// Nothing above reached the child, so the first line it reads is this one.
	if answer := ask(`{"kind":"text","text":"note: fix this","submit":true}`); !answer.OK {
		t.Fatalf("ordinary prose was refused: %s", answer.Reason)
	}
	select {
	case code := <-exited:
		if code != 3 {
			t.Fatalf("exit code %d: something refused above reached the child first", code)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the child never exited")
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
