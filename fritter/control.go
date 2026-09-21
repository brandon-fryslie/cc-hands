package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// What a caller may ask fritter to put into the child's input.
//
// [LAW:types-are-the-program] Exactly one of text and key is meaningful for a given
// kind, and the kind says which, so there is no request that names both and no reader
// that has to guess.
type request struct {
	Kind   string `json:"kind"`   // "text" or "key"
	Text   string `json:"text"`   // kind "text": the characters to type, already escaped by the caller
	Submit bool   `json:"submit"` // kind "text": whether to press Enter after them
	Key    string `json:"key"`    // kind "key": one of the names in keystrokes
}

type response struct {
	OK     bool   `json:"ok"`
	Reason string `json:"reason,omitempty"`
}

// The named chords hands can ask for, and the bytes a terminal sends for each.
//
// This mapping is terminal knowledge and so it lives here rather than in hands' core.
// What a leading slash or at-sign means to Claude Code is the opposite: that is the
// caller's domain, and fritter types the text it is given without re-deciding it
// `[LAW:single-enforcer]`.
var keystrokes = map[string][]byte{
	"escape": {0x1b},
	"enter":  {'\r'},
	"ctrl_c": {0x03},
	// Ctrl-U empties the input box without interrupting, which Ctrl-C only does on the
	// first press - a second press quits the session. It is the chord to reach for when
	// the line has to be cleared and nothing else should happen.
	"ctrl_u":    {0x15},
	"up":        []byte("\x1b[A"),
	"down":      []byte("\x1b[B"),
	"tab":       {'\t'},
	"shift_tab": []byte("\x1b[Z"),
}

// How long one caller may take over the whole of its request and its answer, and the
// most it may send.
//
// [LAW:no-silent-failure] Without a bound, a client that connects and never finishes its
// line holds a goroutine for the rest of the session's life, and the socket is reachable
// by anything running as this user.
const (
	askDeadline = 5 * time.Second
	askLimit    = 64 * 1024
)

// serve answers control connections until the listener is closed.
func (w *Wrapped) serve(listener net.Listener) {
	for {
		connection, err := listener.Accept()
		if err != nil {
			// The listener closed because the child exited; that is the normal way out.
			return
		}
		go w.answer(connection)
	}
}

func (w *Wrapped) answer(connection net.Conn) {
	defer connection.Close()
	if err := connection.SetDeadline(time.Now().Add(askDeadline)); err != nil {
		warn("cannot put a deadline on a control connection: %v", err)
	}
	reader := bufio.NewReader(io.LimitReader(connection, askLimit))
	body, err := reader.ReadBytes('\n')
	if err != nil && len(body) == 0 {
		warn("control connection closed before it asked for anything: %v", err)
		return
	}
	var asked request
	if err := json.Unmarshal(body, &asked); err != nil {
		reply(connection, response{OK: false, Reason: fmt.Sprintf("cannot read the request: %v", err)})
		return
	}
	reply(connection, w.inject(asked))
}

func reply(connection net.Conn, answer response) {
	encoded, err := json.Marshal(answer)
	if err != nil {
		warn("cannot encode the reply %+v: %v", answer, err)
		return
	}
	if _, err := connection.Write(append(encoded, '\n')); err != nil {
		// [LAW:no-silent-failure] The caller is waiting to hear whether the text was
		// typed. A reply that never arrives looks exactly like one that said no.
		warn("cannot answer the control connection: %v", err)
	}
}

// inject puts a request's bytes into the child's input, or says why it did not.
//
// [LAW:no-silent-failure] Every path here either writes the bytes or returns a reason.
// A refused write that reported success would leave hands believing a draft was sent
// when it was not, which is indistinguishable afterwards from one Claude Code ignored.
func (w *Wrapped) inject(asked request) response {
	w.injecting.Lock()
	defer w.injecting.Unlock()

	switch asked.Kind {
	case "text":
		return w.typeText(asked)
	case "key":
		return w.pressKey(asked)
	default:
		return response{OK: false, Reason: fmt.Sprintf("no request kind named %q", asked.Kind)}
	}
}

// typeText is the one request that yields to the person at the keyboard. Text is the only
// thing that can interleave: dropped into a half-written line it produces one prompt made
// of two people's words, which nobody afterwards can pull apart.
func (w *Wrapped) typeText(asked request) response {
	if !w.line.free() {
		return response{OK: false, Reason: "the user has unsent text in this session's input; it is theirs until they submit or cancel it, or until a key request clears the line"}
	}
	// [LAW:no-silent-failure] Bare newlines go to the child as Enter presses, so without
	// bracketing a multi-line draft arrives as several separate submitted prompts. Saying
	// ok to that would tell hands one message was sent when several were.
	if strings.ContainsAny(asked.Text, "\r\n") && !w.paste.enabled() {
		return response{OK: false, Reason: "this session has not turned bracketed paste on, so the newlines in this text would submit it as several separate prompts"}
	}
	if err := w.send(w.paste.encode(asked.Text)); err != nil {
		return response{OK: false, Reason: fmt.Sprintf("nothing was typed: %v", err)}
	}
	if asked.Submit {
		if err := w.send(keystrokes["enter"]); err != nil {
			// A submit is two writes, and the caller has to be able to tell which one
			// failed: retyping text that is already sitting in the box doubles it.
			return response{OK: false, Reason: fmt.Sprintf("the text was typed and is sitting unsent in the input box, but Enter did not land, so do not send it again: %v", err)}
		}
	}
	return response{OK: true}
}

// pressKey sends one chord, and does not yield to the person at the keyboard.
//
// A keystroke cannot interleave with anything: it does exactly what it would do if the
// user had pressed it themselves, and they see the result. Gating it on a free line would
// also be a door locked from the inside - Enter and Ctrl-C are the very keys that free a
// line, so a session whose line is held would have no way back except a human at the
// physical keyboard, which is the case this whole program exists to avoid.
func (w *Wrapped) pressKey(asked request) response {
	chord, known := keystrokes[asked.Key]
	if !known {
		return response{OK: false, Reason: fmt.Sprintf("no key named %q", asked.Key)}
	}
	if err := w.send(chord); err != nil {
		return response{OK: false, Reason: err.Error()}
	}
	w.line.sent(chord)
	return response{OK: true}
}

// send writes to the child without touching the line owner: these bytes are not the
// user's, and counting them as typing would make fritter refuse its own next write.
func (w *Wrapped) send(keys []byte) error {
	if _, err := w.master.Write(keys); err != nil {
		return fmt.Errorf("cannot write to the session: %w", err)
	}
	return nil
}

// control is the socket callers reach this session through, and the private directory it
// lives in.
//
// [LAW:types-are-the-program] The address exists only alongside the listener and the
// directory that holds it, so there is no way to publish an address that nothing is
// listening on and no way to close the socket while leaving its directory behind.
type control struct {
	listener net.Listener
	address  string
	dir      string
}

// listen opens a session's control socket in a directory of its own under parent.
//
// The directory is the session's alone at 0700 and the socket inside it is 0600. Both are
// needed: parent defaults to whatever TMPDIR names, which is private for a login shell but
// is the world-writable /tmp under launchd and cron, and a socket anyone may dial is a
// socket anyone may type `!rm -rf ~` into.
func listen(parent string) (*control, error) {
	dir, err := os.MkdirTemp(parent, "fritter-")
	if err != nil {
		return nil, fmt.Errorf("cannot make a socket directory in %s: %w", parent, err)
	}
	address := filepath.Join(dir, "session.sock")
	// [LAW:no-silent-failure] macOS refuses a unix socket path over 104 bytes with a
	// bind error that names nothing useful, so the length is checked where the path is
	// chosen and reported with the path that was too long.
	if len(address) >= 104 {
		os.RemoveAll(dir)
		return nil, fmt.Errorf("the socket path is %d bytes, over the 104 macOS allows: %s", len(address), address)
	}
	listener, err := net.Listen("unix", address)
	if err != nil {
		os.RemoveAll(dir)
		return nil, fmt.Errorf("cannot listen on %s: %w", address, err)
	}
	if err := os.Chmod(address, 0o600); err != nil {
		listener.Close()
		os.RemoveAll(dir)
		return nil, fmt.Errorf("cannot make the socket %s private: %w", address, err)
	}
	return &control{listener: listener, address: address, dir: dir}, nil
}

// close stops answering and takes the socket and its directory away with it. A socket
// left behind outlives the session it addressed, and the next caller to dial it reaches
// nothing while believing it reached a session.
func (c *control) close() {
	c.listener.Close()
	if err := os.RemoveAll(c.dir); err != nil {
		warn("cannot remove the socket directory %s: %v", c.dir, err)
	}
}
