package main

import (
	"bytes"
	"sync"
)

// The child turns bracketed paste on and off by writing these to its terminal. fritter
// is that terminal, so the mode is a fact it can read rather than one it has to assume.
var (
	pasteOn  = []byte("\x1b[?2004h")
	pasteOff = []byte("\x1b[?2004l")
)

// Wrapping injected text in these tells the child the text was pasted, so a newline
// inside it is a newline in the message rather than the Enter that submits it.
var (
	pasteStart = []byte("\x1b[200~")
	pasteEnd   = []byte("\x1b[201~")
)

// pasteMode watches the child's output for the mode it announces.
//
// [LAW:one-source-of-truth] Whether the child accepts bracketed paste is the child's
// fact. It is read from the stream the child writes, never configured here, so it cannot
// disagree with what the child actually does.
type pasteMode struct {
	mu       sync.Mutex
	on       bool
	pending  []byte // the tail of the last write, in case a sequence is split across reads
	skipping bool   // inside a string sequence, whose contents are text and not commands
}

func newPasteMode() *pasteMode {
	return &pasteMode{}
}

// Write consumes a slice of the child's output. It is an io.Writer so that the same copy
// that reaches the user's screen reaches here, with no second read of the pty.
func (p *pasteMode) Write(output []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// A mode sequence split across two reads would otherwise be missed, so each scan
	// begins with the bytes that could still be its start.
	scan := append(p.pending, output...)
	for i := 0; i < len(scan); {
		// [LAW:one-source-of-truth] The contents of a string sequence - a window title, a
		// tmux passthrough - are text the terminal displays or forwards, not commands it
		// obeys. Matching the mode bytes anywhere would let a title that happens to
		// contain them turn bracketing off on a session that still has it on, and a
		// passthrough turn it on where nothing enabled it. input.go skips these whole
		// when reading the other direction; this reads them the same way.
		if p.skipping {
			end, done := endOfString(scan[i:])
			i += end
			p.skipping = !done
			continue
		}
		if opensString(scan[i:]) {
			p.skipping = true
			i += 2
			continue
		}
		switch {
		case bytes.HasPrefix(scan[i:], pasteOn):
			p.on = true
			i += len(pasteOn)
		case bytes.HasPrefix(scan[i:], pasteOff):
			p.on = false
			i += len(pasteOff)
		default:
			i++
		}
	}
	keep := len(pasteOn) - 1
	if len(scan) < keep {
		keep = len(scan)
	}
	p.pending = append([]byte(nil), scan[len(scan)-keep:]...)

	return len(output), nil
}

// opensString reports whether s begins one of the sequences whose contents are data.
func opensString(s []byte) bool {
	if len(s) < 2 || s[0] != esc {
		return false
	}
	switch s[1] {
	case ']', 'P', '^', '_':
		return true
	}
	return false
}

// endOfString measures how much of s belongs to a string sequence already begun, and
// says whether the sequence ended inside it. They end at BEL or at ESC \.
func endOfString(s []byte) (n int, done bool) {
	for i := 0; i < len(s); i++ {
		if s[i] == 0x07 {
			return i + 1, true
		}
		if s[i] == esc && i+1 < len(s) && s[i+1] == '\\' {
			return i + 2, true
		}
	}
	return len(s), false
}

// enabled reports whether the child currently accepts bracketed paste.
func (p *pasteMode) enabled() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.on
}

// encode renders text as the child should receive it - bracketed when the child asked for
// bracketing, bare when it did not - and says which it did.
//
// [LAW:dataflow-not-control-flow] The mode is read once and the answer carries it out.
// Asking twice - once to decide whether multi-line text is safe, once to encode it - lets
// the child turn bracketing off in between, so a message accepted as one paste goes as
// several submitted prompts.
func (p *pasteMode) encode(text string) (encoded []byte, bracketed bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.on {
		return []byte(text), false
	}
	encoded = make([]byte, 0, len(pasteStart)+len(text)+len(pasteEnd))
	encoded = append(encoded, pasteStart...)
	encoded = append(encoded, text...)
	encoded = append(encoded, pasteEnd...)
	return encoded, true
}
