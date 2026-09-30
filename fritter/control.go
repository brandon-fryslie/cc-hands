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
	Pid     int    `json:"pid"`     // the process the caller means to type into; see inject
	Kind    string `json:"kind"`    // "text", "command" or "key"
	Text    string `json:"text"`    // kinds "text" and "command": the characters to paste, already escaped by the caller
	Command string `json:"command"` // kind "command": the command, typed as keys ahead of its text
	Key     string `json:"key"`     // kind "key": one of the names in keystrokes
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
	"escape":    {0x1b},
	"enter":     {'\r'},
	"ctrl_c":    {0x03},
	"ctrl_u":    {0x15},
	"up":        []byte("\x1b[A"),
	"down":      []byte("\x1b[B"),
	"tab":       {'\t'},
	"shift_tab": []byte("\x1b[Z"),
}

// How long a caller has to send its request, and the most it may send. Without a bound, a
// client that connects and never finishes its line holds a goroutine for the rest of the
// session's life.
const (
	readDeadline  = 1 * time.Second
	askLimit      = 64 * 1024
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
			// accepts connections into a backlog nobody ever reads.
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
	body, err := bufio.NewReader(io.LimitReader(connection, askLimit)).ReadBytes('\n')
	if err != nil {
		reply(connection, refuse(fmt.Sprintf("cannot read the request, which is one JSON object and a newline in at most %d bytes: %v", askLimit, err)))
		return
	}
	var asked request
	if err := json.Unmarshal(body, &asked); err != nil {
		reply(connection, refuse(fmt.Sprintf("cannot read the request: %v", err)))
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
		// [LAW:no-silent-failure] The caller is waiting to hear whether the text was typed.
		warn("cannot answer the control connection: %v", err)
	}
}

func refuse(reason string) response {
	return response{OK: false, Reason: reason}
}

// inject types a request into the child's input, or says why it did not.
func (w *Wrapped) inject(asked request) response {
	// [LAW:single-enforcer] The address reaches a caller by inheritance, and inheritance
	// does not stop at the process fritter wrapped: a second session started from inside
	// this one carries this address as its own. Only fritter knows which process it
	// wrapped, so this is where a request meant for some other session is turned away.
	if child := w.cmd.Process.Pid; asked.Pid != child {
		return refuse(fmt.Sprintf("this socket types into process %d and the request is for process %d. An address inherited from another session reaches that session, not this one", child, asked.Pid))
	}
	keys, err := w.keys(asked)
	if err != nil {
		return refuse(err.Error())
	}
	if err := w.typeIn(keys); err != nil {
		return refuse(fmt.Sprintf("cannot write to the session: %v", err))
	}
	return response{OK: true}
}

// How long keys that end in ESC keep the child's input to themselves. Claude Code 2.1.285
// read an ESC followed within 40ms by another byte as one chord, and alone from 45ms on
// (hands-wire-6ic.ulk, measured 2026-09-30); this is vim's default for the same wait.
const loneEscape = 100 * time.Millisecond

// typeIn writes a request's keys into the child's input.
//
// [LAW:no-ambient-temporal-coupling] A terminal tells an Escape from the start of a chord
// only by the silence after it: an ESC followed at once by Ctrl-C is read as Alt+Ctrl-C,
// and the Escape that was to stop a turn stops nothing. fritter is the one writer into the
// child's input, so it owns that silence: whatever comes next, a request or the user's own
// keys, waits behind the ESC until it has been read alone.
func (w *Wrapped) typeIn(keys []byte) error {
	w.writing.Lock()
	defer w.writing.Unlock()
	if _, err := w.master.Write(keys); err != nil {
		return err
	}
	if keys[len(keys)-1] == esc {
		time.Sleep(loneEscape)
	}
	return nil
}

// keys is a request as the bytes a terminal sends for it.
//
// A text request is the text and a Return, as someone at the keyboard would type them,
// pasted when the child asked for bracketing so that a newline inside it stays a newline.
// It goes as one write: the child reads the paste and the Return behind it together, and
// the Return sends what is in the box.
func (w *Wrapped) keys(asked request) ([]byte, error) {
	switch asked.Kind {
	case "text":
		pasted, err := w.pasted(asked.Text)
		if err != nil {
			return nil, err
		}
		return append(pasted, keystrokes["enter"]...), nil
	case "command":
		// A command is typed, and only its text pasted. A program that folds a long paste
		// into a placeholder folds whatever the paste began with: a command pasted with its
		// text reaches Claude Code as "[Pasted text #1]", which is a prompt, not a command.
		if asked.Command == "" || strings.ContainsAny(asked.Command, " \n") {
			return nil, fmt.Errorf("a command is one word, got %q", asked.Command)
		}
		if offending, at := controlByte(asked.Command); at >= 0 {
			return nil, fmt.Errorf("this command holds the control byte %#02x at offset %d", offending, at)
		}
		if asked.Text == "" {
			return append([]byte(asked.Command), keystrokes["enter"]...), nil
		}
		pasted, err := w.pasted(asked.Text)
		if err != nil {
			return nil, err
		}
		return append(append([]byte(asked.Command+" "), pasted...), keystrokes["enter"]...), nil
	case "key":
		chord, known := keystrokes[asked.Key]
		if !known {
			return nil, fmt.Errorf("no key named %q", asked.Key)
		}
		return chord, nil
	default:
		return nil, fmt.Errorf("no request kind named %q", asked.Kind)
	}
}

// pasted is text as a paste into the child: bracketed when the child asked for it.
func (w *Wrapped) pasted(text string) ([]byte, error) {
	// [LAW:parse-dont-validate] Text is characters and newlines. A control byte in it
	// is a keystroke wearing text's clothes: an ESC ends the bracketing early, and a
	// 0x03 is a Ctrl-C.
	if offending, at := controlByte(text); at >= 0 {
		return nil, fmt.Errorf("this text holds the control byte %#02x at offset %d, which is a keystroke and not a character; send a key request for it", offending, at)
	}
	pasted, bracketed := w.paste.encode(text)
	// [LAW:no-silent-failure] Bare newlines are Returns, so without bracketing a
	// multi-line message arrives as several submitted prompts.
	if !bracketed && strings.Contains(text, "\n") {
		return nil, errors.New("this session has not turned bracketed paste on, so the newlines in this text would submit it as several separate prompts")
	}
	return pasted, nil
}

// controlByte finds the first byte in text that is a keystroke rather than a character,
// and where it is. A newline is neither: bracketing carries it into the message.
func controlByte(text string) (byte, int) {
	for i := 0; i < len(text); i++ {
		if b := text[i]; (b < 0x20 && b != '\n') || b == del {
			return b, i
		}
	}
	return 0, -1
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
