---
description: Turn hands' spoken turn summaries on or off, or say where they stand. Off, a finished turn is not narrated; permission requests, questions asked in a dialog, and plans are always spoken. Takes effect from the next finished turn, with no restart, and lasts across restarts.
argument-hint: "[on|off]"
allowed-tools: Bash(*hands.sessions.summaries*)
---

!`"${CLAUDE_PLUGIN_ROOT}/hooks/python" -m hands.sessions.summaries $ARGUMENTS`

Tell the user, in one line, what the line above says. Do nothing else.
