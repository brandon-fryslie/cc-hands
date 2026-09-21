package main

import (
	"bytes"
	"sync"
)

// Bytes that leave the child's input box empty again: the two that submit a line, and
// the Ctrl-C that clears it.
//
// Escape also clears the box, and is deliberately not here: it is equally the first byte
// of every arrow key, so treating it as a clear would free the line every time the user
// pressed Up. The cost of leaving it out is that a user who presses Escape and walks away
// keeps the line held until they press Enter or Ctrl-C. That is the safe direction to be
// wrong in - a refused write is loud and recoverable, a write into a half-typed line is
// a garbled prompt nobody can attribute.
var lineCleared = []byte{'\r', '\n', 0x03}

// lineOwner answers the one question about the input box that only fritter can answer:
// has the person at the keyboard typed something they have not yet submitted?
//
// hands knows whether a session is Idle, Working or Blocked and decides from that whether
// a send is allowed at all `[LAW:single-enforcer]`. It cannot know this, because the
// keystrokes never reach it. So this is the only input-box fact fritter owns, and it owns
// exactly this much.
type lineOwner struct {
	mu   sync.Mutex
	held bool
}

func newLineOwner() *lineOwner {
	return &lineOwner{}
}

// typed records what the person at the keyboard just sent.
func (l *lineOwner) typed(keystrokes []byte) {
	if len(keystrokes) == 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	// The last clearing byte in the slice decides: a line typed and submitted inside one
	// read leaves the box empty, and one submitted and then typed into again leaves it held.
	if last := bytes.LastIndexAny(keystrokes, string(lineCleared)); last >= 0 {
		l.held = last != len(keystrokes)-1
		return
	}
	l.held = true
}

// free reports whether the input box is clear of the user's own unsent text.
func (l *lineOwner) free() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return !l.held
}
