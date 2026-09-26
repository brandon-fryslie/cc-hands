// Command fritter wraps an interactive terminal program and gives it a second way in.
//
// The program runs on a pseudo-terminal exactly as it would run on yours: same
// interface, same width, same signals, same exit code. Alongside it, fritter listens on
// a unix socket, and anything asked for there is typed into the program's input as
// though someone at the keyboard had typed it.
//
//	fritter [--socket-dir DIR] -- COMMAND [ARGS...]
//
// The socket's path is published to the child in FRITTER_SOCKET, so anything the child
// spawns - a hook, a subprocess - inherits the address and can hand it on. That is the
// whole of fritter's coupling to whatever drives it: one environment variable, carried
// by the operating system along the path that needs it.
//
// fritter knows nothing about Claude Code, and Claude Code is only its first caller.
package main

import (
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
)

const usage = "usage: fritter [--socket-dir DIR] -- COMMAND [ARGS...]"

// Exit codes fritter itself produces. Anything else is the child's own, passed through
// unchanged, because a wrapper that rewrote its child's exit code would make every
// script around it wrong `[LAW:no-silent-failure]`.
const (
	failed = 125 // fritter could not do its job; the child may never have started
	misuse = 64
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout))
}

// run is fritter, with the terminal it uses passed in rather than reached for, so the
// whole of it - the socket taken away on the way out, the child's last output written
// before the process ends - can be run under test as it actually runs.
func run(args []string, stdin *os.File, stdout io.Writer) int {
	// [LAW:no-silent-failure] Armed before there is anything to undo, because the window
	// between creating the socket and forwarding signals is one a termination signal can
	// land in: at its default disposition it kills fritter outright, with the socket left
	// for the next caller to dial into nothing and the user's terminal left in raw mode.
	// A signal arriving before the child exists waits in here and reaches it as soon as
	// the forwarding starts.
	killed := make(chan os.Signal, 1)
	signal.Notify(killed, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP)
	// Stopped last of all, so a signal arriving during the drain, the terminal being put
	// back or the socket being removed is caught rather than ending fritter halfway
	// through them.
	defer signal.Stop(killed)

	dir, argv, err := parse(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fritter: %v\n%s\n", err, usage)
		return misuse
	}

	socket, err := listen(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fritter: %v\n", err)
		return failed
	}
	defer socket.close()

	wrapped, err := start(argv, []string{"FRITTER_SOCKET=" + socket.address})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fritter: %v\n", err)
		return failed
	}
	go wrapped.serve(socket.listener)

	code, err := wrapped.run(stdin, stdout, killed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fritter: %v\n", err)
		return failed
	}
	return code
}

// parse splits fritter's own arguments from the command it wraps at the "--" that
// separates them. The separator is required rather than inferred, so a command with a
// flag fritter also has is never mistaken for fritter's own.
func parse(args []string) (dir string, argv []string, err error) {
	dir = os.TempDir()
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--":
			argv = args[i+1:]
			if len(argv) == 0 {
				return "", nil, fmt.Errorf("no command after --")
			}
			return dir, argv, nil
		case "--socket-dir":
			if i+1 >= len(args) {
				return "", nil, fmt.Errorf("--socket-dir needs a directory")
			}
			dir = args[i+1]
			i++
		default:
			return "", nil, fmt.Errorf("unknown argument %q", args[i])
		}
	}
	return "", nil, fmt.Errorf("no -- separating fritter's arguments from the command")
}

// warn reports something fritter could not do, without stopping what it was doing.
//
// It goes to stderr, which on a wrapped session is the user's own terminal, because a
// wrapper that failed quietly would leave the caller believing text was typed that
// never was `[LAW:no-silent-failure]`.
func warn(format string, args ...any) {
	// A carriage return as well as a line feed, because the terminal this goes to is one
	// fritter put into raw mode, where a bare line feed drops a line without returning to
	// the left margin and every warning after it staircases across the child's interface.
	fmt.Fprintf(os.Stderr, "fritter: "+format+"\r\n", args...)
}
