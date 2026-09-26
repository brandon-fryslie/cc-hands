package main

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// inputQueue is the child's end of the pty's input: the bytes written into the session that
// the child has not read yet.
//
// It exists because the child decides what a read means from the read as a whole, not from
// the keys in it. Claude Code 2.1.283 takes a large enough read as a paste: `a`, Ctrl-S and a
// bracketed paste arriving together had the Ctrl-S stripped as an invisible character, the
// `a` left at the front of the message, and the Enter after them held for review. The same
// three written one after another but read apart did what each says. Writes that follow
// each other closely are read together whenever the child is a moment slow, so no spacing
// of them in time makes them separate reads `[LAW:no-ambient-temporal-coupling]`. What does
// is not writing the next until the child has taken the last - and whether it has is a fact
// the pty keeps: the count of bytes waiting on its slave side.
//
// It has three limits. It is measured on macOS only, and fritter builds nowhere else: Linux
// hands a pty write to the slave on a workqueue, so the count there can read 0 before the
// child can see the bytes at all. A terminal in canonical mode counts
// nothing until a whole line is in, so for a program that reads lines the wait is no wait;
// Claude Code reads raw, where the count is every byte. And a terminal whose session has
// ended cannot be counted at all, so a child that exits on the key it was sent is reported
// as not known to have read it.
type inputQueue struct {
	path string // the slave device, which the child holds open as its terminal
}

// How often the queue is looked at while waiting for the child to read it: at first
// queuePoll, since a running session reads its input at once and this is the whole of what
// the wait usually costs, and then less often, up to queuePollMost. A write given up on is
// waited on for as long as the child does not read, and a stopped session can not read for
// hours.
const (
	queuePoll     = time.Millisecond
	queuePollMost = 100 * time.Millisecond
)

var errQueued = errors.New("the session has not read what is waiting in its input")

// waitEmpty returns once the child has read everything written to it, or errQueued when
// stop fires first. A nil stop never fires.
func (q inputQueue) waitEmpty(stop <-chan time.Time) error {
	// Opened for the wait and not held, because a slave fritter held open would outlive the
	// child, and the master's reader ends only when every slave is closed. O_NOCTTY because
	// a fritter with no terminal of its own would otherwise take this one as its own.
	slave, err := os.OpenFile(q.path, os.O_RDONLY|syscall.O_NOCTTY, 0)
	if err != nil {
		return fmt.Errorf("cannot look at the session's input queue: %w", err)
	}
	defer slave.Close()
	for poll := queuePoll; ; poll = min(2*poll, queuePollMost) {
		waiting, err := unix.IoctlGetInt(int(slave.Fd()), fionread)
		if err != nil {
			return fmt.Errorf("cannot count the session's unread input, so whether it read this is not known - a session that has ended cannot be counted: %w", err)
		}
		if waiting == 0 {
			return nil
		}
		select {
		case <-time.After(poll):
		case <-stop:
			return errQueued
		}
	}
}
