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
// bytes: read as characters, "\x1b[<0;45;12M" is a mouse moving and twelve keys pressed.
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

const (
	esc       = 0x1b
	ctrlC     = 0x03
	backspace = 0x08
	del       = 0x7f
)

// The one report that does not say where it ends: ESC [ M and three raw bytes, any of
// which can be a printable character.
var x10Mouse = []byte("\x1b[M")

// The most bytes held waiting for a sequence to finish. Past this it is not a key: it is
// something answering at length, and none of it is typing.
const carry = 4096

// reader turns the raw bytes arriving on the user's stdin into presses.
//
// It carries state because a terminal's input does not arrive in whole units: a read can
// end in the middle of an escape sequence, and a paste the user made at their own
// keyboard can run across many of them.
type reader struct {
	pending []byte // an unfinished sequence, waiting for the rest of itself
	pasting bool   // inside a paste the user began at their own keyboard
}

// read measures one slice of stdin.
func (r *reader) read(chunk []byte) []press {
	scan := chunk
	if len(r.pending) > 0 {
		scan = append(append([]byte(nil), r.pending...), chunk...)
		r.pending = nil
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
			n, did, complete := r.escape(scan)
			if !complete {
				if len(scan) <= carry {
					r.pending = append([]byte(nil), scan...)
				}
				// Over the limit the bytes are dropped rather than carried or counted.
				// Counted as characters they would hold the line for good, which is the
				// failure this parser exists to end.
				return out
			}
			out = append(out, did)
			scan = scan[n:]
		case b == '\r' || b == '\n' || b == ctrlC:
			out = append(out, press{does: emptied})
			scan = scan[1:]
		case b == del || b == backspace:
			out = append(out, press{does: deleted, count: 1})
			scan = scan[1:]
		case b < 0x20:
			// Every other control byte is a chord that moves the cursor or the history,
			// not one that puts a character in the box.
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
func (r *reader) escape(s []byte) (n int, did press, complete bool) {
	if len(s) < 2 {
		return 0, press{}, false
	}
	switch s[1] {
	case '[':
		return r.csi(s)
	case 'O':
		// SS3: one byte follows. Arrow keys in application mode, and F1 to F4.
		if len(s) < 3 {
			return 0, press{}, false
		}
		return 3, press{}, true
	case ']', 'P', '^', '_':
		// A string sequence, which is how a terminal answers a question at length: the
		// clipboard, its name, its colours. It ends at BEL or at ESC \.
		for i := 2; i < len(s); i++ {
			if s[i] == 0x07 {
				return i + 1, press{}, true
			}
			if s[i] == esc && i+1 < len(s) && s[i+1] == '\\' {
				return i + 2, press{}, true
			}
		}
		return 0, press{}, false
	default:
		// ESC and one more byte is a meta chord, which is a command rather than a
		// character. A lone Escape reaches here only with something after it; by itself
		// it is held as unfinished above, and holding it costs nothing because Escape
		// does not count as a character either way.
		return 2, press{}, true
	}
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
	// Everything else ends at its first final byte, and the parameters before it are
	// skipped rather than read - that skipping is the whole job.
	for i := 2; i < len(s); i++ {
		if s[i] >= 0x40 && s[i] <= 0x7e {
			return i + 1, press{}, true
		}
	}
	return 0, press{}, false
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

// beginningOf reports how many bytes at the end of s could be the start of marker.
func beginningOf(s, marker []byte) int {
	for k := len(marker) - 1; k > 0; k-- {
		if bytes.HasSuffix(s, marker[:k]) {
			return k
		}
	}
	return 0
}
