package main

import "sync"

// How much of the end of the input box is remembered.
//
// One character would answer the only question asked of it - whether the box ends in a
// backslash - but backspacing over the end of a line would then leave the answer unknown,
// and an unknown answer has to be the one that holds the line. A short run keeps the
// ordinary "type it, take some back, submit it" known, and anything longer is a line held
// until the user types again, which is the recoverable direction.
const remembered = 16

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
	// What the box ends with, as far as the bytes say, and whether that is known at all.
	//
	// It exists for one rule in the child. Claude Code 2.1.278 reads a Return that follows
	// a backslash as "keep typing": the backslash becomes a newline and everything already
	// typed stays in the box. Measured against the running program, not assumed - and
	// without it a Return there empties a count that is not empty and hands writes its
	// words into the middle of someone's sentence.
	//
	// Where the cursor is, is not modelled, the same way Ctrl-W's word is not. Moving it
	// and then pressing Return can submit a line this holds, which costs a refusal the
	// user can clear.
	tail  []rune
	murky bool
	// The parser ended a read still holding bytes that might be characters in the box.
	// Kept here rather than asked of the parser, because emptying the box settles it
	// whatever the parser was in the middle of: whatever those bytes were, they are not
	// in the box now.
	unsure bool
}

func newLineOwner() *lineOwner {
	return &lineOwner{}
}

// typed records a slice of the user's stdin, whatever it turns out to hold.
func (l *lineOwner) typed(input []byte) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.fold(l.stdin.read(input))
	// After the fold, not before: bytes left unresolved at the end of this read may be
	// characters typed after anything in it that emptied the box.
	l.unsure = l.stdin.undecided()
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
	if l.fold(chord.read(keys)) {
		// The box is empty, so a sequence the stdin parser had not finished reading is not
		// in it. Kept, those bytes would be read as typing later, and worse, a scan still
		// looking for a terminator would swallow the next Enter along with everything else
		// - so the request sent to free the line would be the reason it stayed held.
		//
		// The sequence only. A paste the user is still making at their own keyboard did
		// not end because the box was emptied, and read outside its brackets the rest of
		// it is typing whose newlines empty a count that is not empty.
		l.stdin.unfinished = partial{}
	}
}

// [LAW:dataflow-not-control-flow] Every press runs the same fold. Where it came from -
// the keyboard or a control request - was decided by which reader produced it, and is not
// a branch here.
func (l *lineOwner) fold(presses []press) (emptiedIt bool) {
	for _, p := range presses {
		switch p.does {
		case inserted:
			for _, r := range string(p.text) {
				l.chars++
				l.tail = append(l.tail, r)
				// Whatever had been forgotten, the box ends with this now.
				l.murky = false
			}
			if len(l.tail) > remembered {
				l.tail = append(l.tail[:0], l.tail[len(l.tail)-remembered:]...)
			}
		case deleted:
			l.chars = max(l.chars-1, 0)
			if len(l.tail) > 0 {
				l.tail = l.tail[:len(l.tail)-1]
			} else {
				l.murky = true
			}
			if l.chars == 0 {
				l.empty()
			}
		case submitted:
			if l.continued() {
				continue
			}
			l.empty()
			emptiedIt = true
		case cancelled:
			// Ctrl-C and Ctrl-U empty the box whatever is in it, which is why they and not
			// Enter are the way back from a line nothing else can settle.
			l.empty()
			emptiedIt = true
		}
	}
	return emptiedIt
}

// continued reports whether this Return went on with the line instead of sending it.
func (l *lineOwner) continued() bool {
	if l.murky {
		// More came out of the box than was being remembered, so what it ends with is not
		// known. A Return that might be a continuation is read as one: a line held after a
		// submit costs a refusal the user's next Enter clears, and a line freed after a
		// continuation is hands typing into a sentence somebody is still writing.
		return true
	}
	if n := len(l.tail); n > 0 && l.tail[n-1] == '\\' {
		// The backslash became the newline. The box is one character different and is not
		// any emptier than it was.
		l.tail[n-1] = '\n'
		return true
	}
	return false
}

// empty records that there is nothing in the box.
func (l *lineOwner) empty() {
	l.chars = 0
	l.tail = l.tail[:0]
	l.murky = false
	// An empty box is empty however unsure the parser was a moment ago. Without this the
	// ctrl_u sent to free a held line frees the count and leaves the doubt, and the line
	// stays held by the very request sent to clear it.
	l.unsure = false
}

// free reports whether the input box is clear of the user's own unsent characters.
//
// A parser that has not finished reading what arrived counts as not clear, because the
// bytes it is still holding may be characters the user typed. Saying yes on a maybe is
// the answer that cannot be taken back.
func (l *lineOwner) free() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.chars == 0 && !l.unsure
}
