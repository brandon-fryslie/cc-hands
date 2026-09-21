package main

import "sync"

// lineOwner answers the one question about the input box that only fritter can answer:
// has the person at the keyboard put characters in it that they have not yet sent?
//
// hands knows whether a session is Idle, Working or Blocked and decides from that whether
// a send is allowed at all `[LAW:single-enforcer]`. It cannot know this, because the
// keystrokes never reach it. So this is the only input-box fact fritter owns, and it owns
// exactly this much.
//
// It is a count of characters rather than a flag, because a text box is a count of
// characters. A flag cannot be told that the user backspaced their way back to empty, so
// it would stay set - and a fact that can only ever become true is not a fact about the
// box, it is a one-way door.
//
// What it cannot see: Ctrl-W and the other chords that take a word or the rest of a line,
// because how many characters they take depends on what was there. Those leave the count
// standing and the line held until the user submits or cancels, or until a key request
// clears it. That is the safe direction to be wrong in - a refused write is loud and
// recoverable, a write into a half-typed line is a garbled prompt nobody can attribute.
type lineOwner struct {
	mu    sync.Mutex
	stdin reader
	chars int
}

func newLineOwner() *lineOwner {
	return &lineOwner{}
}

// typed records a slice of the user's stdin, whatever it turns out to hold.
func (l *lineOwner) typed(input []byte) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.fold(l.stdin.read(input))
}

// sent records what fritter itself put into the box.
//
// A chord fritter sends means what the same chord means when the user presses it - an
// Enter empties the box, an arrow key does not - so it is read by the same parser
// `[LAW:one-source-of-truth]`. It gets a reader of its own because the half-finished
// sequences in the user's stream belong to the user's stream.
func (l *lineOwner) sent(keys []byte) {
	var chord reader
	l.mu.Lock()
	defer l.mu.Unlock()
	l.fold(chord.read(keys))
}

// [LAW:dataflow-not-control-flow] Every press runs the same fold. Where it came from -
// the keyboard or a control request - was decided by which reader produced it, and is not
// a branch here.
func (l *lineOwner) fold(presses []press) {
	for _, p := range presses {
		switch p.does {
		case inserted:
			l.chars += p.count
		case deleted:
			l.chars = max(l.chars-p.count, 0)
		case emptied:
			l.chars = 0
		}
	}
}

// free reports whether the input box is clear of the user's own unsent characters.
//
// A parser that has not finished reading what arrived counts as not clear, because the
// bytes it is still holding may be characters the user typed. Saying yes on a maybe is
// the answer that cannot be taken back.
func (l *lineOwner) free() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.chars == 0 && !l.stdin.undecided()
}
