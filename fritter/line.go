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
// What it cannot see it says it cannot see. Ctrl-W takes a word, Ctrl-U takes a line,
// Ctrl-Y pastes back what was last killed, Tab completes a path, and Up pulls a whole
// previous prompt into a box nothing was typed into - and how much any of that came to is
// not in the bytes. All of them leave the line held until the box is proved empty by a
// Ctrl-C or a Return that really sent. That is the safe direction to be wrong in: a
// refused write is loud and recoverable, a write into a half-typed line is a garbled
// prompt nobody can attribute.
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
	// Where the cursor is, is not modelled - nothing on stdin says. What that costs is a
	// backslash further back in the line than this remembers, with the cursor parked
	// straight after it: the Return that continues it reads as one that sent it. Inside
	// what is remembered, any backslash holds.
	tail  []rune
	murky bool
	// Something changed the box by an amount the bytes did not say: a chord that is not
	// one of the few known to leave it alone, or a history key, which pulls a whole
	// previous prompt into a box that nothing was typed into.
	//
	// Only a box proved empty clears this - a Ctrl-C, or a Return that really did send.
	// [LAW:no-silent-failure] A count that is known to be incomplete is not a count, and
	// reporting it as one is the answer that cannot be taken back.
	disturbed bool
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
			if l.chars == 0 && !l.disturbed {
				l.empty()
			}
		case submitted:
			if l.continued() {
				continue
			}
			// A Return sends whatever is in the box, counted or not, so this is the one
			// thing besides Ctrl-C that settles a box nothing could account for.
			l.empty()
			emptiedIt = true
		case cancelled:
			// One Ctrl-C empties the box however many lines are in it. Measured, and it is
			// why this and not Ctrl-U is the way back from a line nothing else settles.
			// The second press in a row quits the session, so it is sent once.
			l.empty()
			emptiedIt = true
		case disturbed:
			l.disturbed = true
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
	// The child looks at the character before the cursor, wherever the cursor happens to
	// be, and where it is is not something the bytes say. So a backslash anywhere in the
	// remembered end of the line is read as one the cursor might be sitting after.
	// Measured: `ab\c`, one Left, then Return, and the box kept both halves.
	//
	// The last one is the one a Return would have turned into the newline. Turning it
	// keeps the count right and stops this line holding every Return after it.
	for i := len(l.tail) - 1; i >= 0; i-- {
		if l.tail[i] == '\\' {
			l.tail[i] = '\n'
			return true
		}
	}
	return false
}

// empty records that there is nothing in the box.
func (l *lineOwner) empty() {
	l.chars = 0
	l.disturbed = false
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
	return l.chars == 0 && !l.unsure && !l.disturbed
}
