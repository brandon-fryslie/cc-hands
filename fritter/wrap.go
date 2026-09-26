package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

// Wrapped is a child running on a pseudo-terminal that fritter holds the master of.
//
// [LAW:types-are-the-program] A Wrapped exists only once the child is started and the
// pty is open, so there is no "not yet running" wrapper for a caller to write into.
type Wrapped struct {
	master *os.File
	cmd    *exec.Cmd
	paste  *pasteMode
	line   *lineOwner
	// The right to write into the child's input, held by one writer at a time: a request
	// for its check and its one or two writes, the user's keyboard for each read of it, and
	// a write fritter stopped waiting on until the child takes it. Two writers at once put
	// half of each into the input box, and a check that was true before someone else wrote
	// is not true after.
	writing sync.Mutex
	// Set by send, under writing, when it gives a write up and passes the lock to it; see
	// inject, which then leaves the unlocking to that write.
	handedOff bool
}

// start runs argv on a new pty, with env added to the child's environment.
//
// The pty is the whole point: Claude Code asks whether its input is a terminal and takes
// a different, non-interactive path when it is not. A wrapper built on pipes would run a
// different program than the one the user is trying to drive.
func start(argv []string, env []string) (*Wrapped, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = append(os.Environ(), env...)
	master, err := pty.Start(cmd)
	if err != nil {
		return nil, fmt.Errorf("cannot start %s on a pty: %w", argv[0], err)
	}
	return &Wrapped{master: master, cmd: cmd, paste: newPasteMode(), line: newLineOwner()}, nil
}

// run pumps the terminal and the child into each other until the child exits, and
// returns the child's exit code.
//
// stdin must be the terminal, because raw mode and the window size are set on it.
// stdout is only ever written to, so it is asked for no more than that.
//
// [LAW:dataflow-not-control-flow] Both directions run the same copy every iteration.
// Where a byte came from - the user's keyboard or the control socket - is a value the
// writer carries, never a branch in the pump.
func (w *Wrapped) run(stdin *os.File, stdout io.Writer, killed <-chan os.Signal) (int, error) {
	// Closed last of everything here, so the forwarding below covers the drain and the
	// terminal being put back, and then ends: the channel is the caller's and outlives run.
	finished := make(chan struct{})
	defer close(finished)
	// The session is over, so its input is too. Left open, the master outlives run with
	// the stdin reader and any request still in flight writing into a pty nobody reads;
	// closed, they are told so. Deferred before the restore, so it runs after it: the
	// resizer reads the master until the restore has joined it.
	defer func() {
		if err := w.master.Close(); err != nil {
			warn("cannot close the session's pty: %v", err)
		}
	}()
	restore, err := w.attach(stdin)
	if err != nil {
		return 0, err
	}
	// [LAW:no-silent-failure] A terminal left in raw mode is a shell the user cannot
	// type into afterwards, so the restore runs on every path out of here, including a
	// panic, and says so if it cannot.
	defer restore()

	// The child's output is read here rather than copied blind, because the mode it
	// announces - bracketed paste on or off - is carried in it and nothing else reports it.
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		_, _ = io.Copy(io.MultiWriter(stdout, w.paste), w.master)
	}()
	// [LAW:no-silent-failure] cmd.Wait returns the moment the child is reaped, which is
	// before the last of what it printed has been copied out - `fritter -- sh -c 'echo
	// done'` can otherwise print nothing at all. Waiting here, on every path out and
	// before the terminal is restored under the copy, is what makes the output whole.
	//
	// The wait is bounded because the pty can be held open by something the child left
	// running, and an unbounded wait would keep fritter alive after its session ended.
	// Reaching the bound means output really was lost, so it is said out loud.
	defer func() {
		select {
		case <-drained:
		case <-time.After(drainGrace):
			warn("the session's output was still arriving %s after it exited; the end of it is lost", drainGrace)
		}
	}()
	// This read outlives run, and deliberately so. main hands it os.Stdin, and a read
	// already blocked there cannot be interrupted: SetReadDeadline answers "file type does
	// not support deadline" for a terminal, so joining on it would hang the exit instead of
	// hurrying it. Measured, because an earlier version of this comment had it backwards: a
	// pty slave does take a deadline and it does unblock a read in flight; os.Stdin, /dev/tty
	// and the pty master do not. The slave is the shape the tests run in, not the shape a
	// session runs in, and a contract that holds only under test is not one.
	// [LAW:no-silent-failure] A guarantee that deadlocks is worse than one
	// not made, so the contract is stated rather than faked: run returns when the child
	// exits, this goroutine may still be blocked reading stdin, and it touches nothing
	// but stdin and the pty. The caller must not close stdin before its process ends -
	// which for fritter is the next statement in main, so there is no window.
	go func() {
		_, _ = io.Copy(w, stdin)
	}()

	// [LAW:no-silent-failure] A fritter killed from outside would otherwise die with its
	// defers unrun: the socket left behind for the next caller to dial into nothing, and
	// the user's terminal left in raw mode. Forwarding instead ends the child, which ends
	// the wait below, which runs every cleanup on the ordinary path out.
	//
	// The channel is armed by the caller, before the socket or the raw terminal exist, so
	// that the window before this goroutine starts is covered too.
	//
	// Ctrl-C at the keyboard never arrives here: in raw mode it is byte 0x03 travelling to
	// the child through the pty, which is what makes it the child's interrupt and not ours.
	go func() {
		for {
			select {
			case received := <-killed:
				if err := w.cmd.Process.Signal(received); err != nil {
					warn("cannot pass %s to the session: %v", received, err)
				}
			case <-finished:
				return
			}
		}
	}()

	if err := w.cmd.Wait(); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return exitCode(exit), nil
		}
		return 0, fmt.Errorf("waiting for %s: %w", w.cmd.Path, err)
	}
	return 0, nil
}

// How long run waits for the child's last output after the child is gone.
const drainGrace = 2 * time.Second

// Write forwards what arrived on the user's stdin to the child, and lets the line owner
// read it for the one fact about the input box that only fritter knows.
//
// Not everything here is typing. A terminal in raw mode also answers the child's own
// questions on this same stream, so the line owner parses rather than counts - see
// input.go, which exists because counting was wrong.
func (w *Wrapped) Write(input []byte) (int, error) {
	// Waited for rather than refused: these are the user's own keys, and nothing may drop
	// them. The wait is behind one request's writes, or behind a child that is reading
	// nothing, in which case these would have blocked in the pty just the same.
	w.writing.Lock()
	defer w.writing.Unlock()
	w.line.typed(input)
	return w.master.Write(input)
}

// attach puts the real terminal into raw mode, matches the pty's window to it, and keeps
// them matched. The returned function undoes all of it.
func (w *Wrapped) attach(tty *os.File) (func(), error) {
	state, err := term.MakeRaw(int(tty.Fd()))
	if err != nil {
		return nil, fmt.Errorf("cannot put the terminal into raw mode: %w", err)
	}
	// The child draws a full-screen interface, so a pty that does not match the real
	// window renders to the wrong width. Set it once now and again on every resize.
	resized := make(chan os.Signal, 1)
	signal.Notify(resized, syscall.SIGWINCH)
	resizing := make(chan struct{})
	go func() {
		defer close(resizing)
		for range resized {
			if err := pty.InheritSize(tty, w.master); err != nil {
				warn("cannot resize the pty: %v", err)
			}
		}
	}()
	resized <- syscall.SIGWINCH

	return func() {
		signal.Stop(resized)
		close(resized)
		// Joined before the restore below touches the same terminal, so no resize is
		// still reading the descriptor when the caller is free to close it.
		<-resizing
		if err := term.Restore(int(tty.Fd()), state); err != nil {
			warn("cannot restore the terminal: %v", err)
		}
	}, nil
}

// exitCode is what the child's fate looks like to whatever ran fritter.
//
// A child killed by a signal has no exit code of its own, and ExitCode answers -1 for
// it. Passing that on would exit 255, which is a real code some programs use and would
// tell a caller that fritter's child exited with 255 rather than that it was killed.
// Shells report a signalled child as 128 plus the signal, and everything that reads exit
// codes already knows that convention, so fritter speaks it too.
// [LAW:no-silent-failure] The alternative is a wrapper that quietly rewrites how its
// child died.
func exitCode(exit *exec.ExitError) int {
	if status, ok := exit.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		return 128 + int(status.Signal())
	}
	return exit.ExitCode()
}
