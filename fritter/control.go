package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
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
	"escape":    {0x1b},
	"enter":     {'\r'},
	"ctrl_c":    {0x03},
	"up":        []byte("\x1b[A"),
	"down":      []byte("\x1b[B"),
	"tab":       {'\t'},
	"shift_tab": []byte("\x1b[Z"),
}

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
	reader := bufio.NewReader(connection)
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

	// The person at the keyboard outranks hands, for every kind of input alike: typing
	// into their half-written line garbles it, and Up or Ctrl-C would throw it away.
	if !w.line.free() {
		return response{OK: false, Reason: "the user has unsent text in this session's input; it is theirs until they press Enter or Ctrl-C"}
	}

	switch asked.Kind {
	case "text":
		// Bracketed when the child asked for bracketing, so a newline inside the text is
		// a newline in the message and not the Enter that submits a half-written one.
		if err := w.send(w.paste.encode(asked.Text)); err != nil {
			return response{OK: false, Reason: err.Error()}
		}
		if asked.Submit {
			if err := w.send(keystrokes["enter"]); err != nil {
				return response{OK: false, Reason: err.Error()}
			}
		}
		return response{OK: true}
	case "key":
		chord, known := keystrokes[asked.Key]
		if !known {
			return response{OK: false, Reason: fmt.Sprintf("no key named %q", asked.Key)}
		}
		if err := w.send(chord); err != nil {
			return response{OK: false, Reason: err.Error()}
		}
		return response{OK: true}
	default:
		return response{OK: false, Reason: fmt.Sprintf("no request kind named %q", asked.Kind)}
	}
}

// send writes to the child without touching the line owner: these bytes are not the
// user's, and counting them as typing would make fritter refuse its own next write.
func (w *Wrapped) send(keys []byte) error {
	if _, err := w.master.Write(keys); err != nil {
		return fmt.Errorf("cannot write to the session: %w", err)
	}
	return nil
}

// listen opens the control socket in dir and returns it with the address to publish.
func listen(dir string) (net.Listener, string, error) {
	socket, err := os.CreateTemp(dir, "fritter-*.sock")
	if err != nil {
		return nil, "", fmt.Errorf("cannot make a socket path in %s: %w", dir, err)
	}
	address := socket.Name()
	socket.Close()
	// CreateTemp made the path to reserve the name; the listener needs it free.
	if err := os.Remove(address); err != nil {
		return nil, "", fmt.Errorf("cannot clear the socket path %s: %w", address, err)
	}
	// [LAW:no-silent-failure] macOS refuses a unix socket path over 104 bytes with a
	// bind error that names nothing useful, so the length is checked where the path is
	// chosen and reported with the path that was too long.
	if len(address) >= 104 {
		return nil, "", fmt.Errorf("the socket path is %d bytes, over the 104 macOS allows: %s", len(address), address)
	}
	listener, err := net.Listen("unix", address)
	if err != nil {
		return nil, "", fmt.Errorf("cannot listen on %s: %w", address, err)
	}
	return listener, address, nil
}
