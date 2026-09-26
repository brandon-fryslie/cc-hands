package main

// FIONREAD, _IOR('f', 127, int): how many bytes are waiting to be read on a descriptor.
// x/sys/unix does not export it for darwin.
const fionread = 0x4004667f
