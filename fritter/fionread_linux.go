package main

import "golang.org/x/sys/unix"

// How many bytes are waiting to be read on a descriptor.
const fionread = unix.TIOCINQ
