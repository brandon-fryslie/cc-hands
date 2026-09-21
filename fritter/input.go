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
	does  effect
	count int // inserted and deleted: how many characters
}

type effect int

const (
	nothing  effect = iota // a report, or a key that moves the cursor rather than the text
	inserted               // characters went into the box
	deleted                // characters came out of it
	emptied                // the box is empty now: the line was submitted or cancelled
)

// The bytes that leave the box empty, and the ones that take a character out of it.
//
// Measured against Claude Code 2.1.278 rather than assumed: Ctrl-C and Ctrl-U both empty
// the box and Escape does not touch it, which is the opposite of what this file used to
// say. Ctrl-W takes the last word and is not here, because how many characters that is
// depends on what the word was - it leaves the count standing, which holds the line and
// is the safe direction.
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
	pending []byte // an unfinished sequence, waiting for the rest of itself
	alone   bool   // the pending bytes are an ESC that arrived with nothing after it
	pasting bool   // inside a paste the user began at their own keyboard
}

// read measures one slice of stdin.
func (r *reader) read(chunk []byte) []press {
	scan := chunk
	// An ESC held over from the last read arrived with nothing after it. The bytes of one
	// keypress are written by the terminal together and arrive together, so whatever
	// follows in this read belongs to a new keypress. That is the only evidence there is
	// for telling an Escape the user pressed from the ESC of a sequence a read happened
	// to cut in half, and settling says which sequence gets to use it.
	settling := r.alone
	if len(r.pending) > 0 {
		scan = append(append([]byte(nil), r.pending...), chunk...)
		r.pending = nil
		r.alone = false
	}

	var out []press
	for len(scan) > 0 {
		if r.pasting {
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
				r.pending = append([]byte(nil), scan...)
				r.alone = len(scan) == 1
				return out
			}
			out = append(out, did)
			scan = scan[n:]
		case b == '\r' || b == '\n' || b == ctrlC || b == ctrlU:
			out = append(out, press{does: emptied})
			scan = scan[1:]
		case b == del || b == backspace:
			out = append(out, press{does: deleted, count: 1})
			scan = scan[1:]
		case b < 0x20:
			// Every other control byte is a chord that moves the cursor, the history or a
			// word, not one that puts a character in the box.
			out = append(out, press{does: nothing})
			scan = scan[1:]
		default:
			n, chars, whole := printable(scan)
			out = append(out, press{does: inserted, count: chars})
			if !whole {
				// A read can end mid-character, and half a character is not one yet.
				r.pending = append([]byte(nil), scan[n:]...)
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
	switch next := s[1]; {
	case next == esc:
		// Two escapes running are not one sequence. Taking both would throw away the
		// introducer of whatever the second one begins and leave its parameters to be
		// read as typing - which is how a mouse report becomes twelve keypresses. Take
		// the first alone; the second starts again from here.
		return 1, press{}, true
	case settling && (next < 0x20 || next == del):
		// No sequence has a control byte in second place, and this ESC came in a read of
		// its own, so it was the Escape key and this is the next thing the user pressed.
		// Take the ESC alone and read on: what follows may be the Enter that empties the
		// box, and swallowing it would hold a line that is no longer there.
		return 1, press{}, true
	case next == '[':
		return r.csi(s)
	case next == 'O':
		// SS3: one byte follows. Arrow keys in application mode, and F1 to F4.
		if len(s) < 3 {
			return 0, press{}, false
		}
		return 3, press{}, true
	case next == ']' || next == 'P' || next == '^' || next == '_':
		// A string sequence, which is how a terminal answers a question at length: the
		// clipboard, its name, its colours. It ends at BEL or at ESC \.
		for i := 2; i < len(s) && i < sequenceLimit; i++ {
			if s[i] == 0x07 {
				return i + 1, press{}, true
			}
			if s[i] == esc && i+1 < len(s) && s[i+1] == '\\' {
				return i + 2, press{}, true
			}
			// A terminal's answer is a printable payload. A control byte inside one says
			// this was never a sequence, and waiting for a terminator that is not coming
			// would swallow the Enter the user just pressed along with everything else.
			if s[i] < 0x20 || s[i] == del {
				return 1, press{}, true
			}
		}
		return unterminated(s)
	default:
		// ESC and one more byte, arriving together, is a meta chord - Alt and a key. It
		// is a command rather than a character, except for the Option-Enter that puts a
		// newline in the box; counting that as nothing leaves the line held, which is the
		// safe direction, and the user's next Enter clears it.
		return 2, press{}, true
	}
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
		r.pasting = true
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
			return i + 1, press{}, true
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
		r.pasting = false
		return end + len(pasteEnd), press{does: inserted, count: utf8.RuneCount(s[:end])}, true
	}
	// The end marker can straddle a read, so a tail that could be its beginning is held
	// back rather than counted as text.
	held := beginningOf(s, pasteEnd)
	r.pending = append([]byte(nil), s[len(s)-held:]...)
	return len(s), press{does: inserted, count: utf8.RuneCount(s[:len(s)-held])}, false
}

// printable measures the run of characters at the front of s, stopping at anything that
// is not one. It reports the bytes consumed, the characters they spell, and whether the
// run ended on a whole character.
func printable(s []byte) (n, chars int, whole bool) {
	for n < len(s) {
		switch b := s[n]; {
		case b < 0x20 || b == del:
			return n, chars, true
		case b < utf8.RuneSelf:
			n++
		default:
			if !utf8.FullRune(s[n:]) {
				return n, chars, false
			}
			_, size := utf8.DecodeRune(s[n:])
			n += size
		}
		chars++
	}
	return n, chars, true
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
	return len(r.pending) > 1
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
