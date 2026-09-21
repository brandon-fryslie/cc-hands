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
	// One injection at a time: a request is a check and one or two writes, and two
	// interleaving would put half of each into the input box.
	injecting sync.Mutex
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
func (w *Wrapped) run(stdin *os.File, stdout io.Writer) (int, error) {
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
	go func() {
		_, _ = io.Copy(io.MultiWriter(stdout, w.paste), w.master)
	}()
	// This read outlives run, and deliberately so. A read already blocked on a terminal
	// cannot be interrupted portably: SetReadDeadline answers "file type does not support
	// deadline" for a pty slave on macOS, and where it does answer nil it does not reliably
	// unblock a read already in flight, so joining on it would hang the exit instead of
	// hurrying it. [LAW:no-silent-failure] A guarantee that deadlocks is worse than one
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
	// Ctrl-C at the keyboard never arrives here: in raw mode it is byte 0x03 travelling to
	// the child through the pty, which is what makes it the child's interrupt and not ours.
	killed := make(chan os.Signal, 1)
	signal.Notify(killed, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	defer signal.Stop(killed)
	go func() {
		for received := range killed {
			if err := w.cmd.Process.Signal(received); err != nil {
				warn("cannot pass %s to the session: %v", received, err)
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

// Write forwards the user's own keystrokes to the child and notes that they now hold a
// partly-typed line, which is the one fact about the input box that only fritter knows.
func (w *Wrapped) Write(keystrokes []byte) (int, error) {
	w.line.typed(keystrokes)
	return w.master.Write(keystrokes)
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
