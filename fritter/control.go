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
	Pid    int    `json:"pid"`    // the process the caller means to type into; see inject
	Kind   string `json:"kind"`   // "text" or "key"
	Text   string `json:"text"`   // kind "text": the characters to type, already escaped by the caller
	Submit bool   `json:"submit"` // kind "text": whether to press Enter after them
	Key    string `json:"key"`    // kind "key": one of the names in keystrokes
}

type response struct {
	OK     bool   `json:"ok"`
	Reason string `json:"reason,omitempty"`
	// What a refusal leaves of the request in the session. Empty only when OK.
	Typed typed `json:"typed,omitempty"`
}

// typed is what of a refused request may be in the session: none of it, or perhaps some or
// all of it.
//
// [LAW:types-are-the-program] The reason says the same thing in English, and a person reads
// it. A caller acts on it: resending a request of which nothing was typed is safe, and
// resending one that may be in the box sends it twice. That decision cannot rest on the
// wording of a sentence, so it travels as a value beside it.
type typed string

const (
	typedNothing typed = "nothing"
	typedMaybe   typed = "maybe"
)

// refuse is the one way to say no, so no refusal can leave out what it left behind.
func refuse(left typed, reason string) response {
	return response{OK: false, Reason: reason, Typed: left}
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
	// Ctrl-U kills back to the start of the line the cursor is on, and that is the line as
	// displayed: measured, a 250-character prompt in a 100-column terminal lost one row to
	// a single press and kept 192 characters. It is offered because a caller may want it,
	// but it empties nothing. Text is typed into a box fritter has emptied itself; see stash.
	"ctrl_u":    {0x15},
	"up":        []byte("\x1b[A"),
	"down":      []byte("\x1b[B"),
	"tab":       {'\t'},
	"shift_tab": []byte("\x1b[Z"),
}

// emptying leaves the input box empty and Claude Code's stash with nothing in it, whatever
// either held before. Each is one step, read on its own.
//
// Ctrl-S in Claude Code 2.1.283 moves the box into the stash - every line of it, wherever
// the cursor is, in shell mode or out, with a completion list open or not - and leaves the
// box empty in prompt mode. Into an empty box it does the opposite and puts the stash back.
// So:
//
//   - `a`, Ctrl-S: the box is certain not to be empty when the Ctrl-S lands, so it is
//     stashed, and the box is empty. The `a` has to be a character the box keeps: a space
//     or a no-break space typed into an empty box leaves it reading as empty, and the
//     Ctrl-S then restored the stash under whatever came next.
//   - `a`, Ctrl-S again: the stash is now exactly `a`. Whatever it held is gone; the stash
//     is fritter's to use.
//   - Ctrl-S into the empty box: the `a` comes back, with the cursor after it, and the
//     stash is empty.
//   - Backspace: takes the character before the cursor, which is that `a`.
//
// The stash has to end empty because the session puts it back into the box as it sends
// a prompt, and it does that some time after reading the Enter: a stash written a
// millisecond after the Enter was read put the text away unsent. With nothing stashed,
// nothing comes back, and nothing has to follow the Enter.
//
// Ctrl-C is not a way to do it. Into an idle session it empties the box and arms the next
// press to quit; into a working one it stops the work and leaves the box as it was - and
// which of the two the session is, stdin does not say.
var emptying = [][]byte{[]byte("a\x13"), []byte("a\x13"), {0x13}, {del}}

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
//	readDeadline (1s) + claimGrace (0.5s) + typeGrace (2s) + replyDeadline (1s)
//	  = 4.5s < hands' ANSWER_TIMEOUT (5s)
const (
	readDeadline  = 1 * time.Second
	replyDeadline = 1 * time.Second
	askLimit      = 64 * 1024
)

// How long a request's writes into the child's input may take, all of them together,
// before fritter stops waiting on them, and how long serve waits before accepting again
// after an error it did not expect.
//
// Every write waits for the child to read it (see inputQueue), which a running session
// does at once and a stopped or wedged one never does. A wait with no bound there is the
// whole program hanging on the one thing it exists to do, so the wait ends and says so
// instead. One budget for the request rather than one per write, because a text request
// is six writes and the sum above has to hold for all of them.
const (
	claimGrace    = 500 * time.Millisecond
	typeGrace     = 2 * time.Second
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
			reply(connection, refuse(typedNothing, fmt.Sprintf("the request passed %d bytes without ending in a newline", askLimit)))
			return
		}
		if len(body) == 0 {
			warn("control connection closed before it asked for anything: %v", err)
			return
		}
	}
	var asked request
	if err := json.Unmarshal(body, &asked); err != nil {
		reply(connection, refuse(typedNothing, fmt.Sprintf("cannot read the request: %v", err)))
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
	// [LAW:single-enforcer] The address reaches a caller by inheritance, and inheritance
	// does not stop at the process fritter wrapped: a second session started from inside
	// this one - from its shell, or from a tmux server first started there - carries this
	// address as its own. Only fritter knows which process it wrapped, so this is where a
	// request meant for some other session is turned away, before a byte of it is typed
	// into this one.
	if child := w.cmd.Process.Pid; asked.Pid != child {
		return refuse(typedNothing, fmt.Sprintf("this socket types into process %d and the request is for process %d; nothing was typed. An address inherited from another session reaches that session, not this one", child, asked.Pid))
	}
	// [LAW:no-ambient-temporal-coupling] Waited for, but not for long. The user's keys and
	// the terminal's reports hold the input for an instant each, and a request arriving
	// in one of those instants should not be turned away for it. Anything holding it
	// longer is another request's writes or a write the child is not reading, which ends
	// when it ends - and the wait is spent out of the caller's budget, so it is bounded.
	select {
	case w.writing <- struct{}{}:
	case <-time.After(claimGrace):
		return refuse(typedNothing, fmt.Sprintf("another write into this session has not finished after %s - another request's, or an earlier one the child is not reading - so nothing was typed", claimGrace))
	}
	claim := hold{by: time.After(typeGrace)}
	answer := w.dispatch(asked, &claim)
	if !claim.passed {
		<-w.writing
	}
	return answer
}

// hold is one request's claim on w.writing, and whether it has been passed on.
//
// [LAW:no-shared-mutable-globals] Local to the request that took the lock. On Wrapped it
// would be read after the lock had been passed to a write and let go, by which time the
// next request may have taken the lock and reset it - and the first would unlock the
// second's claim.
type hold struct {
	// A write this request gave up on has the lock now and lets go of it when the child
	// takes it; see send.
	passed bool
	// When the request's writes stop being waited on.
	by <-chan time.Time
}

func (w *Wrapped) dispatch(asked request, claim *hold) response {
	switch asked.Kind {
	case "text":
		return w.typeText(asked, claim)
	case "key":
		return w.pressKey(asked, claim)
	default:
		return refuse(typedNothing, fmt.Sprintf("no request kind named %q", asked.Kind))
	}
}

// typeText types text into a box it has emptied first.
//
// Emptying it is not a courtesy to the person at the keyboard but the only way the text
// arrives as itself: typed onto whatever they had half-written, it would be one prompt made
// of two people's words, which nobody afterwards can pull apart. What they had written is
// not kept; see emptying.
func (w *Wrapped) typeText(asked request, claim *hold) response {
	// [LAW:parse-dont-validate] Text is characters and newlines. A control byte in it is
	// a keystroke wearing text's clothes: an ESC ends the bracketing early and everything
	// after it is typed and submitted on its own, and a 0x03 is a Ctrl-C. Either way what
	// the caller asked to send as one message arrives as something else, under an ok.
	if offending, at := controlByte(asked.Text); at >= 0 {
		return refuse(typedNothing, fmt.Sprintf("this text holds the control byte %#02x at offset %d, which is a keystroke and not a character; send a key request for it", offending, at))
	}
	body, end, bracketed := w.paste.encode(asked.Text)
	// [LAW:no-silent-failure] Bare newlines go to the child as Enter presses, so without
	// bracketing a multi-line draft arrives as several separate submitted prompts. Saying
	// ok to that would tell hands one message was sent when several were.
	if !bracketed && strings.Contains(asked.Text, "\n") {
		return refuse(typedNothing, "this session has not turned bracketed paste on, so the newlines in this text would submit it as several separate prompts")
	}
	// [LAW:no-silent-failure] An Enter the child takes as something other than a send
	// leaves the text in the box, and an ok would tell the caller it was sent.
	if asked.Submit && staysUnsent(asked.Text) {
		return refuse(typedNothing, "this text ends where the session takes Enter as something other than sending - after a backslash, which becomes a newline, or on an @, # or : token, whose completion list takes the Enter - so it would sit unsent; nothing was typed. A space after the token closes the list")
	}
	return w.write(claim, textSteps(body, end, asked.Submit))
}

// textSteps is a text request as the writes it is made of, in order.
//
// [LAW:dataflow-not-control-flow] A request is its steps, and each is written the same
// way. What differs is what a failure at that step leaves behind, which the caller has
// to be told: retyping text that is already in the box doubles it.
//
// The Enter goes in with the marker that closes the paste. Read apart from it, even a
// millisecond later, the session took the Enter before the paste was in the box and
// sent nothing.
func textSteps(body, end []byte, submit bool) []step {
	var steps []step
	for _, keys := range emptying {
		steps = append(steps, step{keys, "the input box was being emptied, so what it and the stash held may be gone, and none of the text was typed: %s", typedNothing, typedNothing})
	}
	closing := "the text is in the input box, unsent, and whether what closes it was read is not known; the next text request empties the box: %s"
	if submit {
		end = append(end, keystrokes["enter"]...)
		// Whether the Enter was read is exactly what an unknowable end cannot say, so the
		// caller is told it may have been sent - and a session that quit on it cannot be asked.
		closing = "the text is in the input box and whether what closes it was read is not known, so it may have been sent; do not send it again: %s"
	}
	// The text is the first step whose keys are the request's own; once it is all in, what
	// follows it can only fail with the text already in the box.
	return append(steps, step{body, "%s", typedMaybe, typedNothing}, step{end, closing, typedMaybe, typedMaybe})
}

// step is one write of a request, and what to tell the caller if it does not land.
type step struct {
	keys   []byte
	failed string // a format taking what went wrong with the write
	// What of the request a failure at this step leaves in the session: when some of this
	// step's keys may have reached it, and when none did.
	touched, untouched typed
}

// left is what of the request a failure at this step leaves in the session.
func (s step) left(d delivery) typed {
	if d.touched() {
		return s.touched
	}
	return s.untouched
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

// pressKey sends one chord. It does exactly what it would do if the user had pressed it
// themselves, and they see the result.
func (w *Wrapped) pressKey(asked request, claim *hold) response {
	chord, known := keystrokes[asked.Key]
	if !known {
		return refuse(typedNothing, fmt.Sprintf("no key named %q", asked.Key))
	}
	return w.write(claim, []step{keyStep(chord)})
}

// keyStep is a key request's one write: all of it is the request's own.
func keyStep(chord []byte) step {
	return step{chord, "%s", typedMaybe, typedNothing}
}

// write sends a request's steps in order, each once the child has read the one before, and
// stops at the first that does not land.
//
// The child is waited on once before the first step as well: what the user typed a moment
// before this request took the lock may still be unread, and read together with the first
// step it is one read the child can take for a paste (see inputQueue). Between steps there is
// nothing to wait for twice - each step's own wait saw the queue empty, and the lock keeps
// every other writer out, the user's keyboard included.
func (w *Wrapped) write(claim *hold, steps []step) response {
	if err := w.queue.waitEmpty(claim.by); err != nil {
		return refuse(typedNothing, fmt.Sprintf("nothing was typed: %v", err))
	}
	for _, s := range steps {
		landed := w.send(claim, s.keys)
		if wrong, bad := landed.wrong(); bad {
			return refuse(s.left(landed), fmt.Sprintf(s.failed, wrong))
		}
	}
	return response{OK: true}
}

// landing is how a write into the session ended.
type landing int

const (
	arrived    landing = iota // every byte reached the child, and the child read it
	partway                   // the write came back with an error, so how much landed is known
	unknowable                // the child had not read it in time, so how much landed cannot be known
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
		return fmt.Sprintf("%d of %d bytes reached the input box before the write failed, so what is there is a fragment, which the next text request empties: %v", d.landed, d.of, d.why), true
	default:
		return d.why.Error(), true
	}
}

// touched says whether any of the write may have reached the session: some bytes did, or
// how many did is not known.
func (d delivery) touched() bool {
	return d.how == unknowable || d.landed > 0
}

// send writes to the child, waits for the child to read it, and reports what became of it.
//
// The write runs on a goroutine because neither it nor the wait after it has a deadline to
// set: a pty master is not a file the runtime can poll, so SetWriteDeadline answers "file
// type does not support deadline", and the only bound available is to stop waiting. Neither
// is cancelled - they cannot be - so while one is outstanding nothing else may write, or
// what the next request writes is read together with these bytes. So a write given up on
// keeps the right to write: the lock passes to it, and it lets go the moment the child has
// read what it was holding.
//
// Called only under a request's claim on w.writing, which is what it passes on.
func (w *Wrapped) send(claim *hold, keys []byte) delivery {
	type outcome struct {
		n           int
		wrote, read error
	}
	done := make(chan outcome, 1)
	go func() {
		n, err := w.master.Write(keys)
		finished := outcome{n: n, wrote: err}
		if err == nil {
			finished.read = w.queue.waitEmpty(nil)
		}
		done <- finished
	}()
	select {
	case finished := <-done:
		switch {
		case finished.wrote != nil:
			return delivery{how: partway, landed: finished.n, of: len(keys), why: fmt.Errorf("cannot write to the session: %w", finished.wrote)}
		case finished.read != nil:
			return delivery{how: unknowable, of: len(keys), why: finished.read}
		}
		return delivery{how: arrived, landed: finished.n, of: len(keys)}
	case <-claim.by:
		claim.passed = true
		go func() {
			<-done
			<-w.writing
		}()
		return delivery{how: unknowable, of: len(keys), why: fmt.Errorf("the session had not read this request's keys %s after it was asked, so it is not reading its input or is reading it slowly; how much of this write landed is not known", typeGrace)}
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
