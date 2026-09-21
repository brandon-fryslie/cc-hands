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
	mu      sync.Mutex
	on      bool
	pending []byte // the tail of the last write, in case a sequence is split across reads
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
	for i := 0; i < len(scan); i++ {
		switch {
		case bytes.HasPrefix(scan[i:], pasteOn):
			p.on = true
		case bytes.HasPrefix(scan[i:], pasteOff):
			p.on = false
		}
	}
	keep := len(pasteOn) - 1
	if len(scan) < keep {
		keep = len(scan)
	}
	p.pending = append([]byte(nil), scan[len(scan)-keep:]...)

	return len(output), nil
}

// enabled reports whether the child currently accepts bracketed paste.
func (p *pasteMode) enabled() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.on
}

// encode renders text as the child should receive it: bracketed when the child asked for
// bracketing, bare when it did not.
//
// [LAW:dataflow-not-control-flow] The caller always calls encode and always writes what
// it returns. The mode changes the bytes, not which code runs.
func (p *pasteMode) encode(text string) []byte {
	if !p.enabled() {
		return []byte(text)
	}
	encoded := make([]byte, 0, len(pasteStart)+len(text)+len(pasteEnd))
	encoded = append(encoded, pasteStart...)
	encoded = append(encoded, text...)
	encoded = append(encoded, pasteEnd...)
	return encoded
}
