package main

import (
	"bufio"
	"encoding/json"
	"errors"
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

// How long each phase of one exchange may take, and the most a caller may send.
//
// [LAW:no-silent-failure] Without a bound, a client that connects and never finishes its
// line holds a goroutine for the rest of the session's life, and the socket is reachable
// by anything running as this user.
//
// [LAW:one-source-of-truth] The bounds are per phase rather than one over the whole
// connection, and that is the point rather than an accident. Injecting is the slow phase:
// under a single deadline a write that spends nearly all of it leaves nothing for the
// reply, so fritter types the text and is then unable to say that it did. A caller told
// only that the connection closed reads it as "nothing happened" and sends the message
// again. The reply is the one thing that must always have time left, so it is given its
// own budget after the typing is over.
//
// Their sum is what hands must outlast, and does:
//
//	readDeadline (1s) + at most two writes at writeGrace (2s) + replyDeadline (1s)
//	  = 4s < hands' ANSWER_TIMEOUT (5s)
const (
	readDeadline  = 1 * time.Second
	replyDeadline = 1 * time.Second
	askLimit      = 64 * 1024
)

// How long a write into the child's input may take before fritter stops waiting on it,
// and how long serve waits before accepting again after an error it did not expect.
//
// A pty in raw mode holds a kilobyte of input, and a write that fills it blocks until the
// child reads - which a running session does at once, and a stopped or wedged one never
// does. Measured on macOS: a cooked pty takes 300KB without blocking, a raw one blocks at
// 1024 bytes, and Claude Code runs raw. A wait with no bound there is the whole program
// hanging on the one thing it exists to do, so the wait ends and says so instead.
const (
	writeGrace    = 1 * time.Second
	acceptBackoff = 50 * time.Millisecond
)

// serve answers control connections until the listener is closed.
func (w *Wrapped) serve(listener net.Listener) {
	for {
		connection, err := listener.Accept()
		if err != nil {
			// [LAW:no-silent-failure] Only a closed listener ends this loop. Returning on
			// any error at all would end it on a passing one - running out of file
			// descriptors, say - and leave the session running with a socket that still
			// accepts connections into a backlog nobody ever reads. A caller would dial
			// it successfully and then wait forever.
			if errors.Is(err, net.ErrClosed) {
				return
			}
			warn("cannot take a control connection: %v", err)
			time.Sleep(acceptBackoff)
			continue
		}
		go w.answer(connection)
	}
}

func (w *Wrapped) answer(connection net.Conn) {
	defer connection.Close()
	if err := connection.SetReadDeadline(time.Now().Add(readDeadline)); err != nil {
		warn("cannot put a deadline on a control connection: %v", err)
	}
	reader := bufio.NewReader(io.LimitReader(connection, askLimit))
	body, err := reader.ReadBytes('\n')
	if err != nil {
		// [LAW:no-silent-failure] Hitting the cap has to say so. Left to fall through, a
		// request cut off mid-way is refused for the JSON it no longer ends with, which
		// tells a caller its draft was malformed when what happened is that it was long.
		if int64(len(body)) >= askLimit {
			reply(connection, response{OK: false, Reason: fmt.Sprintf("the request passed %d bytes without ending in a newline", askLimit)})
			return
		}
		if len(body) == 0 {
			warn("control connection closed before it asked for anything: %v", err)
			return
		}
	}
	var asked request
	if err := json.Unmarshal(body, &asked); err != nil {
		reply(connection, response{OK: false, Reason: fmt.Sprintf("cannot read the request: %v", err)})
		return
	}
	reply(connection, w.inject(asked))
}

func reply(connection net.Conn, answer response) {
	// Set here, after any injecting is done, and so unspent by it. A caller is waiting to
	// hear whether its text was typed, and that answer has to be affordable even when the
	// typing took everything the request itself was allowed.
	if err := connection.SetWriteDeadline(time.Now().Add(replyDeadline)); err != nil {
		warn("cannot put a deadline on a reply: %v", err)
	}
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
	// [LAW:parse-dont-validate] Text is characters and newlines. A control byte in it is
	// a keystroke wearing text's clothes: an ESC ends the bracketing early and everything
	// after it is typed and submitted on its own, and a 0x03 is a Ctrl-C. Either way what
	// the caller asked to send as one message arrives as something else, under an ok.
	if offending, at := controlByte(asked.Text); at >= 0 {
		return response{OK: false, Reason: fmt.Sprintf("this text holds the control byte %#02x at offset %d, which is a keystroke and not a character; send a key request for it", offending, at)}
	}
	encoded, bracketed := w.paste.encode(asked.Text)
	// [LAW:no-silent-failure] Bare newlines go to the child as Enter presses, so without
	// bracketing a multi-line draft arrives as several separate submitted prompts. Saying
	// ok to that would tell hands one message was sent when several were.
	if !bracketed && strings.Contains(asked.Text, "\n") {
		return response{OK: false, Reason: "this session has not turned bracketed paste on, so the newlines in this text would submit it as several separate prompts"}
	}
	if wrong, bad := w.send(encoded).wrong(); bad {
		return response{OK: false, Reason: wrong}
	}
	if asked.Submit {
		if wrong, bad := w.send(keystrokes["enter"]).wrong(); bad {
			// A submit is two writes, and the caller has to be able to tell which one
			// failed: retyping text that is already sitting in the box doubles it.
			return response{OK: false, Reason: fmt.Sprintf("the text was typed and is sitting unsent in the input box, but Enter did not land, so do not send it again: %s", wrong)}
		}
	}
	return response{OK: true}
}

// controlByte finds the first byte in text that is a keystroke rather than a character,
// and where it is. A newline is neither: bracketing carries it into the message, and
// typeText refuses it in its own right when the session will not bracket.
func controlByte(text string) (byte, int) {
	for i := 0; i < len(text); i++ {
		if b := text[i]; (b < 0x20 && b != '\n') || b == del {
			return b, i
		}
	}
	return 0, -1
}

// pressKey sends one chord, and does not yield to the person at the keyboard.
//
// A keystroke cannot interleave with anything: it does exactly what it would do if the
// user had pressed it themselves, and they see the result. Gating it on a free line would
// also be a door locked from the inside - Enter, Ctrl-C and Ctrl-U are the very keys that
// free a line, so a session whose line is held would have no way back except a human at
// the physical keyboard, which is the case this whole program exists to avoid.
func (w *Wrapped) pressKey(asked request) response {
	chord, known := keystrokes[asked.Key]
	if !known {
		return response{OK: false, Reason: fmt.Sprintf("no key named %q", asked.Key)}
	}
	if wrong, bad := w.send(chord).wrong(); bad {
		// Recorded only for a chord that went out whole: half of one is not a chord the
		// child acted on, and crediting it would free a line that is still held.
		return response{OK: false, Reason: wrong}
	}
	w.line.sent(chord)
	return response{OK: true}
}

// landing is how a write into the session ended.
type landing int

const (
	arrived    landing = iota // every byte reached the input box
	partway                   // the write came back with an error, so how much landed is known
	unknowable                // the write never came back, so how much landed cannot be known
)

// delivery is what became of one write into the session.
//
// [LAW:types-are-the-program] A count and an error cannot say "unknown", and a write that
// had to be given up on is exactly that: the bytes sit in a queue the child has not read,
// and nothing on this side can see how far down they went. Forced into a count it comes
// out zero, and zero is rendered "nothing was typed" - the one thing that must never be
// said about a message half of which is already in front of the user.
type delivery struct {
	how    landing
	landed int   // how many bytes are known to have reached the box
	of     int   // how many were asked for
	why    error // nil only when how is arrived
}

// wrong says what to tell the caller about a write, and reports false when there is
// nothing to tell because it landed.
//
// [LAW:one-source-of-truth] Every refusal about a write is worded here, so a text request
// and a key request cannot describe the same outcome in two different ways, and no caller
// can reach for "nothing was typed" over an outcome that does not know.
func (d delivery) wrong() (string, bool) {
	switch d.how {
	case arrived:
		return "", false
	case partway:
		// [LAW:no-silent-failure] A write that failed after some bytes left them in the
		// input box, and "nothing was typed" would send a caller to retype a message half
		// of which is already there.
		if d.landed == 0 {
			return fmt.Sprintf("nothing was typed: %v", d.why), true
		}
		return fmt.Sprintf("%d of %d bytes reached the input box before the write failed, so what is there is a fragment; clear the line before sending anything else: %v", d.landed, d.of, d.why), true
	default:
		return d.why.Error(), true
	}
}

// send writes to the child and reports what became of it.
//
// It does not touch the line owner: these bytes are not the user's, and counting them as
// typing would make fritter refuse its own next write.
//
// The write runs on a goroutine because a pty write has no deadline to set: a pty master
// is not a file the runtime can poll, so SetWriteDeadline answers "file type does not
// support deadline" and the only bound available is to stop waiting. The write itself is
// not cancelled - it cannot be - so while one is outstanding nothing else may write, or
// two half-written messages interleave into one nobody can attribute. It clears itself
// the moment the child starts reading again.
func (w *Wrapped) send(keys []byte) delivery {
	if w.stuck.Load() {
		return delivery{how: partway, of: len(keys), why: errors.New("an earlier write to this session has not finished, so the child is not reading its input")}
	}
	type written struct {
		n   int
		err error
	}
	done := make(chan written, 1)
	w.stuck.Store(true)
	go func() {
		n, err := w.master.Write(keys)
		w.stuck.Store(false)
		done <- written{n, err}
	}()
	select {
	case landed := <-done:
		if landed.err != nil {
			return delivery{how: partway, landed: landed.n, of: len(keys), why: fmt.Errorf("cannot write to the session: %w", landed.err)}
		}
		return delivery{how: arrived, landed: landed.n, of: len(keys)}
	case <-time.After(writeGrace):
		return delivery{how: unknowable, of: len(keys), why: fmt.Errorf("the session did not take this within %s, so it is not reading its input; how much of it landed is not known", writeGrace)}
	}
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
