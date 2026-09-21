package main

import (
	"bytes"
	"unicode/utf8"
)

// What one piece of the user's input does to the child's input box.
//
// [LAW:parse-dont-validate] The bytes arriving on stdin are not a stream of characters.
// A terminal in raw mode also sends the child's own questions back answered - focus
// reports, mouse reports, cursor positions - and those answers are full of printable
// bytes: read as characters, "\x1b[<0;45;12M" is a mouse moving and ten keys pressed.
// So the bytes are turned into presses once, here, and nothing downstream sees a byte.
type press struct {
	does effect
	text []byte // inserted: the characters that went in, borrowed until the fold reads them
}

type effect int

const (
	nothing   effect = iota // a report, or a key that leaves the box exactly as it was
	inserted                // characters went into the box
	deleted                 // one character came out of it
	submitted               // Return, which empties the box - unless what it follows says otherwise
	cancelled               // Ctrl-C, which empties it whatever is in it
	killed                  // Ctrl-U, which empties the line the cursor is on and not the box
	disturbed               // the box changed by some amount these bytes do not say
)

// The chords that leave the box exactly as it was.
//
// [LAW:parse-dont-validate] Everything else that is not a character is read as disturbed,
// and this list is short on purpose. The ways to edit a text box are many and belong to
// the child rather than to fritter - Ctrl-Y pastes back whatever was last killed, Ctrl-R
// searches the history into the box, Tab completes a path into it, Ctrl-G opens an editor
// on it - and every one of them that is guessed harmless and is not frees a line that is
// not free. Listing the harmless ones and disturbing by default is the only way round
// that does not need the list to be complete.
var cursorOnly = map[byte]bool{
	0x01: true, // Ctrl-A, to the start of the line
	0x02: true, // Ctrl-B, back one character
	0x05: true, // Ctrl-E, to the end of the line
	0x06: true, // Ctrl-F, forward one character
	0x0c: true, // Ctrl-L, redraw
}

// The bytes that leave the box empty, and the ones that take a character out of it.
//
// Measured against Claude Code 2.1.278 rather than assumed, and more than once, because
// the first measurements were wrong. One Ctrl-C empties the box however many lines are in
// it. Ctrl-U does not: it kills the line the cursor is on, so a box of three lines took
// four presses and still had its first line. Escape does not touch the box at all.
// Ctrl-W takes the last word, and how many characters that is depends on the word.
const (
	esc       = 0x1b
	ctrlC     = 0x03
	ctrlU     = 0x15
	backspace = 0x08
	del       = 0x7f
)

// The one report that does not say where it ends: ESC [ M and three raw bytes, any of
// which can be a printable character.
var x10Mouse = []byte("\x1b[M")

// The most bytes a sequence may run to before the parser stops believing it is one.
//
// This bound is what stands between a lone Escape and the rest of the session. An ESC is
// both the Escape key and the first byte of every arrow key, mouse report and terminal
// answer, and the bytes alone do not say which. Press Escape and then type `Please fix
// it` and those bytes read as the opening of a device control string: unbounded, the scan
// for its terminator swallows every character the user types and never finds one. Past
// the bound the ESC is taken for the Escape key it was and what follows is read as what
// it is, so the parser always comes back into step on its own.
//
// It is small on purpose. Every real sequence a terminal sends in answer is well inside
// it - the longest measured here is a colour reply at 27 bytes - and the cost of a bound
// too large is a line held until the user's next Enter.
const sequenceLimit = 64

// reader turns the raw bytes arriving on the user's stdin into presses.
//
// It carries state because a terminal's input does not arrive in whole units: a read can
// end in the middle of an escape sequence, and a paste the user made at their own
// keyboard can run across many of them.
type reader struct {
	unfinished partial  // a sequence, or a character, that has not finished arriving
	paste      pasteRun // the paste the user is making at their own keyboard
}

// partial is a sequence, or a character, that a read ended in the middle of.
//
// [LAW:types-are-the-program] It is a value of its own because emptying the input box
// invalidates exactly this much and no more: whatever these bytes were going to turn out
// to be, they are not in the box any longer.
type partial struct {
	bytes []byte // what has arrived of it
	alone bool   // it is a single ESC, which arrived with nothing after it
}

// pasteRun is a paste the user began at their own keyboard, and the bytes at the end of
// the last read that could be the beginning of the marker that ends it.
//
// [LAW:types-are-the-program] Those held bytes live here rather than in partial because
// an emptied box settles a half-read sequence and settles nothing about a paste: the
// terminal sends the rest of one regardless. Sharing one slot, "forget the sequence" and
// "forget the marker" are the same statement - and forgetting the marker leaves the
// parser inside a paste that has ended, counting every Enter afterwards as a character,
// which no keystroke recovers from.
type pasteRun struct {
	on   bool
	tail []byte
}

// read measures one slice of stdin.
func (r *reader) read(chunk []byte) []press {
	scan := chunk
	// An ESC held over from the last read arrived with nothing after it. The bytes of one
	// keypress are written by the terminal together and arrive together, so whatever
	// follows in this read belongs to a new keypress. That is the only evidence there is
	// for telling an Escape the user pressed from the ESC of a sequence a read happened
	// to cut in half, and settling says which sequence gets to use it.
	settling := r.unfinished.alone
	switch {
	case len(r.paste.tail) > 0:
		scan = append(append([]byte(nil), r.paste.tail...), chunk...)
		r.paste.tail = nil
	case len(r.unfinished.bytes) > 0:
		scan = append(append([]byte(nil), r.unfinished.bytes...), chunk...)
		r.unfinished = partial{}
	}

	var out []press
	for len(scan) > 0 {
		if r.paste.on {
			n, did, ended := r.inPaste(scan)
			out = append(out, did)
			if !ended {
				return out
			}
			scan = scan[n:]
			continue
		}
		switch b := scan[0]; {
		case b == esc:
			n, did, complete := r.escape(scan, settling)
			settling = false
			if !complete {
				// The rest of it has not arrived. It is held rather than guessed at, and
				// undecided below is what the line owner reads while it waits - these
				// bytes may yet turn out to be the user's.
				r.unfinished = partial{bytes: append([]byte(nil), scan...), alone: len(scan) == 1}
				return out
			}
			out = append(out, did)
			scan = scan[n:]
		case b == '\r':
			// What a Return does to the box depends on what is in the box, which is not
			// something the bytes on stdin can say `[LAW:one-source-of-truth]`. The press
			// names the key and the box decides the effect.
			out = append(out, press{does: submitted})
			scan = scan[1:]
		case b == '\n':
			// Not the same key, and not the same thing. A bare 0x0A is Ctrl-J, which
			// Claude Code's own footer offers as the way to put a newline in a prompt -
			// it is the multi-line prompt for every terminal that cannot send Shift-Enter.
			// The character goes into the box and nothing is sent. Measured: after Ctrl-J
			// the prompt was still there and a send landed underneath it.
			//
			// Return reaches a raw terminal as 0x0D, so nothing here loses a submit.
			out = append(out, press{does: inserted, text: scan[:1]})
			scan = scan[1:]
		case b == ctrlC:
			out = append(out, press{does: cancelled})
			scan = scan[1:]
		case b == ctrlU:
			// Not the same as Ctrl-C. Measured: on a box of three lines it took four
			// presses and the first line was still there, because it kills the line the
			// cursor is on rather than the box. The box says what that came to.
			out = append(out, press{does: killed})
			scan = scan[1:]
		case b == del || b == backspace:
			out = append(out, press{does: deleted})
			scan = scan[1:]
		case b < 0x20:
			out = append(out, press{does: chord(b)})
			scan = scan[1:]
		default:
			n, whole := printable(scan)
			out = append(out, press{does: inserted, text: scan[:n]})
			if !whole {
				// A read can end mid-character, and half a character is not one yet.
				r.unfinished = partial{bytes: append([]byte(nil), scan[n:]...)}
				return out
			}
			scan = scan[n:]
		}
	}
	return out
}

// escape measures the sequence at the front of s and says what it did to the box. It
// reports false when s ends before the sequence does.
func (r *reader) escape(s []byte, settling bool) (n int, did press, complete bool) {
	if len(s) < 2 {
		return 0, press{}, false
	}
	if settling {
		// This ESC arrived in a read of its own. The bytes of one keypress are written by
		// the terminal together, so nothing arriving in a later read belongs to it: it was
		// the Escape key, and what follows is the next thing the user pressed.
		//
		// That holds for `[` and `O` too, which are the second byte of an arrow key and
		// also two characters people type. The bytes cannot say which, so what is chosen
		// here is the direction to be wrong in. Read as an arrow key, an Escape and a
		// typed `Ok` leave the count at zero with two characters in the box, free reports
		// the line clear, and hands writes over the user's words - nothing undoes that.
		// Read as typing, an arrow key whose sequence really was split counts two
		// characters that are not there and holds a line that is empty, which the user's
		// next Enter clears and which hands can clear itself with a ctrl_c, because a key
		// is never refused. One of those is recoverable.
		return 1, press{}, true
	}
	switch next := s[1]; {
	case next == esc:
		// Two escapes running are not one sequence. Taking both would throw away the
		// introducer of whatever the second one begins and leave its parameters to be
		// read as typing - which is how a mouse report becomes ten keypresses. Take
		// the first alone; the second starts again from here.
		return 1, press{}, true
	case next == '[':
		return r.csi(s)
	case next == 'O':
		// SS3: one byte follows. Arrow keys in application mode, and F1 to F4.
		if len(s) < 3 {
			return 0, press{}, false
		}
		if s[2] < 0x20 || s[2] == del {
			// The same rule csi and the string scan carry: no SS3 ends on a control byte,
			// so this was never one, and taking three bytes regardless would swallow the
			// Enter the user just pressed.
			return 1, press{}, true
		}
		return 3, press{does: cursorKey(s[2])}, true
	case next == ']' || next == 'P' || next == '^' || next == '_':
		// A string sequence, which is how a terminal answers a question at length: the
		// clipboard, its name, its colours. It ends at BEL or at ESC \.
		for i := 2; i < len(s) && i < sequenceLimit; i++ {
			if s[i] == 0x07 {
				return i + 1, press{}, true
			}
			if s[i] == esc {
				// The ESC of the ST terminator, decided before the control-byte rule
				// below can mistake it for one. A DCS reply is always ST-terminated -
				// BEL is not legal for it - so reading the ESC as proof the sequence
				// never was one would read every answer to XTGETTCAP as typing.
				if i+1 >= len(s) {
					// ST cut in half by the end of the read. The rest is coming.
					return unterminated(s)
				}
				if s[i+1] == '\\' {
					return i + 2, press{}, true
				}
				return 1, press{}, true
			}
			// A terminal's answer is a printable payload. A control byte inside one says
			// this was never a sequence, and waiting for a terminator that is not coming
			// would swallow the Enter the user just pressed along with everything else.
			if s[i] < 0x20 || s[i] == del {
				return 1, press{}, true
			}
		}
		return unterminated(s)
	case next < 0x20 || next == del:
		// ESC and a control byte together is a meta chord, and the ones that matter change
		// the box by amounts these two bytes do not say: Option-Enter puts a newline in it,
		// Option-Backspace takes a word out. Reading the second byte on its own would be
		// worse still - the Return of an Option-Enter would empty a count that is not empty.
		return 2, press{does: disturbed}, true
	default:
		// ESC and an ordinary character together: Alt and a letter, or the Escape key and
		// the letter after it. One read is no proof of one keypress - over ssh or through
		// tmux, everything typed within a round trip arrives together - so this is the
		// same ambiguity as above and gets the same answer. A chord read as two keypresses
		// holds a line that may be empty; two keypresses read as a chord lose a character
		// off the count and free a line holding one.
		return 1, press{}, true
	}
}

// chord says what a control byte did to the box.
func chord(b byte) effect {
	if cursorOnly[b] {
		return nothing
	}
	return disturbed
}

// cursorKey says what a sequence ending in this byte did to the box.
//
// Almost every one of them is the terminal answering a question or the cursor moving, and
// neither touches the text: focus reports end in I or O, mouse reports in M or m, a cursor
// position in R, a device attributes reply in c. The ones that do touch it are the history
// keys - Up and Down pull a whole previous prompt in, which is how an empty box fills
// while nothing is typed - and the `~` forms, Delete among them.
func cursorKey(final byte) effect {
	switch final {
	case 'A', 'B', '~':
		return disturbed
	}
	return nothing
}

// unterminated is what to do with a sequence that has not ended yet: wait for the rest of
// it, or, past the bound, decide it was never a sequence and give back the ESC alone so
// the loop reads the rest as ordinary input.
func unterminated(s []byte) (n int, did press, complete bool) {
	if len(s) >= sequenceLimit {
		return 1, press{}, true
	}
	return 0, press{}, false
}

func (r *reader) csi(s []byte) (n int, did press, complete bool) {
	// The same markers the child is sent when fritter injects text, arriving the other
	// way: this is the user pasting at their own keyboard.
	if bytes.HasPrefix(s, pasteStart) {
		r.paste.on = true
		return len(pasteStart), press{}, true
	}
	if bytes.HasPrefix(s, x10Mouse) {
		if len(s) < len(x10Mouse)+3 {
			return 0, press{}, false
		}
		return len(x10Mouse) + 3, press{}, true
	}
	// Everything else ends at its first final byte, and the parameters and intermediates
	// before it are skipped rather than read - that skipping is the whole job. All of them
	// are printable, so a control byte here says this was never a sequence either.
	for i := 2; i < len(s) && i < sequenceLimit; i++ {
		if s[i] >= 0x40 && s[i] <= 0x7e {
			return i + 1, press{does: cursorKey(s[i])}, true
		}
		if s[i] < 0x20 || s[i] == del {
			return 1, press{}, true
		}
	}
	return unterminated(s)
}

// inPaste measures the part of s inside a paste the user began at their own keyboard.
// Everything up to the end marker is characters going into the box - newlines included,
// which is what the markers are for.
func (r *reader) inPaste(s []byte) (n int, did press, ended bool) {
	if end := bytes.Index(s, pasteEnd); end >= 0 {
		r.paste.on = false
		return end + len(pasteEnd), press{does: inserted, text: s[:end]}, true
	}
	// The end marker can straddle a read, so a tail that could be its beginning is held
	// back rather than counted as text.
	held := beginningOf(s, pasteEnd)
	r.paste.tail = append([]byte(nil), s[len(s)-held:]...)
	return len(s), press{does: inserted, text: s[:len(s)-held]}, false
}

// printable measures the run of characters at the front of s, stopping at anything that
// is not one. It reports the bytes consumed and whether the run ended on a whole
// character. What those bytes spell is the box's business, not this function's.
func printable(s []byte) (n int, whole bool) {
	for n < len(s) {
		switch b := s[n]; {
		case b < 0x20 || b == del:
			return n, true
		case b < utf8.RuneSelf:
			n++
		default:
			if !utf8.FullRune(s[n:]) {
				return n, false
			}
			_, size := utf8.DecodeRune(s[n:])
			n += size
		}
	}
	return n, true
}

// undecided reports whether the parser is holding bytes that could be characters the user
// has put in the box.
//
// A lone ESC is not. Whether it opened a sequence or was the Escape key, it puts nothing
// in the box, and the next read says which it was. From the second byte on the answer is
// yes: a scan still looking for its terminator is holding bytes that may turn out to be
// typing, and reporting an empty box while it does is the one mistake with no recovery -
// hands writes its own words into a line that already has the user's.
//
// [LAW:no-silent-failure] An ambiguity the bytes cannot settle is reported as one rather
// than guessed at. It settles itself on the next read, or at sequenceLimit.
func (r *reader) undecided() bool {
	return len(r.unfinished.bytes) > 1 || len(r.paste.tail) > 0
}

// beginningOf reports how many bytes at the end of s could be the start of marker.
func beginningOf(s, marker []byte) int {
	for k := len(marker) - 1; k > 0; k-- {
		if bytes.HasSuffix(s, marker[:k]) {
			return k
		}
	}
	return 0
}
