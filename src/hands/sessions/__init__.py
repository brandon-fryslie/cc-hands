"""Every edge on the Claude Code side: hook shims, the hook socket, session files, transcripts.

Kept free of imports here, because the shim runs from this package in Claude
Code's critical path and must not pay for the daemon's dependencies.
"""
