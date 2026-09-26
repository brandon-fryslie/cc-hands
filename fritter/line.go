package main

import (
	"regexp"
	"strings"
	"sync"
)

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
// It is one fact and not a count. A count was the obvious shape - a text box is a
// number of characters - and it was wrong, because nothing on stdin ever takes the
// number back down. Backspace looked like the exception and is not: the child reads it
// as `if(this.isAtStart())return this`, so a Backspace with the cursor at the start of
// the box removes nothing, and the cursor is not something stdin says. A count that only
// ever rises is a flag that has learnt to add.
//
// What it cannot see it says it cannot see. Ctrl-W takes a word, Ctrl-U takes a line,
// Ctrl-Y pastes back what was last killed, Backspace takes one character or none, Tab
// completes a path, and Up pulls a whole previous prompt into a box nothing was typed
// into - and how much any of that came to is not in the bytes. All of them leave the line
// held until the box is proved empty by a Ctrl-C or a Return that really sent. That is
// the safe direction to be wrong in: a refused write is loud and recoverable, and a key
// request is never refused, so hands can always clear a line this holds too long. A write
// into a half-typed line is a garbled prompt nobody can attribute.
type lineOwner struct {
	mu    sync.Mutex
	stdin reader
	// Something is in the box that was not seen to leave it: characters that went in, or
	// a key whose effect on the box the bytes do not say. The two were separate fields
	// until Backspace stopped being countable and left them with the same life - set by
	// what the user does, cleared only by a box proved empty `[LAW:one-type-per-behavior]`.
	//
	// [LAW:no-silent-failure] A box that is known to hold something unaccounted for is not
	// an empty box, and reporting it as one is the answer that cannot be taken back.
	held bool
	// What the box ends with, as far as the bytes say.
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
	tail []rune
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
			l.held = true
			for _, r := range string(p.text) {
				l.tail = append(l.tail, r)
			}
			if len(l.tail) > remembered {
				l.tail = append(l.tail[:0], l.tail[len(l.tail)-remembered:]...)
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
			//
			// It is also the one thing left here that is trusted and should not be. That
			// measurement was taken on an idle child; a Ctrl-C while the child is working
			// interrupts the work and leaves the box exactly as it was, so this empties a
			// box that still holds the user's next prompt. Tracked as
			// hands-harness-5nb.dh7. Whether the child is working is not on stdin - the
			// Return that starts it is seen and its end never is - so it wants a second
			// source of truth rather than another rule in here.
			l.empty()
			emptiedIt = true
		case disturbed:
			l.held = true
		}
	}
	return emptiedIt
}

// continued reports whether this Return went on with the line instead of sending it.
//
// Two things in the child take a Return and do something with it other than send. Both
// are decided against the cursor, which stdin does not say, so both are read off the
// remembered end of the line and both are read the way that holds.
func (l *lineOwner) continued() bool {
	return l.escaping() || l.completing()
}

// escaping reports whether the Return followed a backslash, which the child turns into a
// newline and keeps typing.
func (l *lineOwner) escaping() bool {
	// The child looks at the character before the cursor, wherever the cursor happens to
	// be, and where it is is not something the bytes say. So a backslash anywhere in the
	// remembered end of the line is read as one the cursor might be sitting after.
	// Measured: `ab\c`, one Left, then Return, and the box kept both halves.
	//
	// The last one is the one a Return would have turned into the newline. Turning it
	// here keeps this line from holding every Return after it, and it is what the child
	// did to the box, so the remembered end stays a true copy of it.
	for i := len(l.tail) - 1; i >= 0; i-- {
		if l.tail[i] == '\\' {
			l.tail[i] = '\n'
			return true
		}
	}
	return false
}

// completing reports whether a completion list could be open over the box, in which case
// the Return picked an entry out of it and sent nothing.
//
// Claude Code's input box answers a Return in two quite different ways, and which one it
// takes is not in the keystroke. With no suggestions up it submits. With suggestions up
// it calls `preventDefault()` and applies the highlighted entry instead, which leaves the
// box *longer* than it was and still unsent. An `@` naming a file is how prompts point at
// code, and a directory keeps the list up for the press after, so this is a key people
// hold down, not a corner.
//
// Whether the list is up is decided by the token ending at the cursor. Neither the token
// nor the cursor is visible here, so what is asked instead is whether any cursor position
// inside the remembered end of the line would have opened one - the same question the
// child asks, over the part of the answer this can see. What came before the remembered
// end is not remembered, and a space is one of the things it could have been, which is
// what the `^` in the patterns reads it as.
func (l *lineOwner) completing() bool {
	for cursor := range l.tail {
		if opensList(string(l.tail[:cursor+1])) {
			return true
		}
	}
	return false
}

// opensList reports whether a cursor at the end of text would have a completion list open.
//
// [LAW:one-source-of-truth] The child's own patterns, read out of 2.1.278 rather than
// guessed, and joined into one:
//
//	@ /(^|[\s\u3002\u3001\uFF1F\uFF01])@([\p{L}\p{N}\p{M}_\-./\\()[\]~:]*|"[^"]*"?)$/u
//	# /(^|\s)#([a-z0-9][a-z0-9_-]*)$/
//	: /(^|\s):([a-z0-9_+-]{2,})$/
//
// The `*` on the first is why `@` needs nothing after it: the cursor sitting straight
// after an `@` already opens the list on every file there is.
//
// A slash command is not here. Its Return runs the command and empties the box - the
// child passes `shouldExecute` true on that path - so it is an ordinary submit.
func opensList(text string) bool {
	return listToken.MatchString(text)
}

var listToken = regexp.MustCompile(`(?:^|[\s\x{3002}\x{3001}\x{FF1F}\x{FF01}])@(?:[\p{L}\p{N}\p{M}_\-./\\()\[\]~:]*|"[^"]*"?)$` +
	`|(?:^|\s)#[a-z0-9][a-z0-9_-]*$` +
	`|(?:^|\s):[a-z0-9_+-]{2,}$`)

// staysUnsent reports whether an Enter pressed straight after text, with the cursor at its
// end, would go on with the line instead of sending it: after a backslash, which the child
// turns into a newline, or under a completion list, which takes the Enter for itself.
//
// It is `continued` with the cursor known. Text fritter types into an empty box leaves the
// cursor at its end, so here the two rules can be asked exactly rather than read the way
// that holds.
func staysUnsent(text string) bool {
	return strings.HasSuffix(text, `\`) || opensList(text)
}

// empty records that there is nothing in the box.
func (l *lineOwner) empty() {
	l.held = false
	l.tail = l.tail[:0]
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
	return !l.held && !l.unsure
}
