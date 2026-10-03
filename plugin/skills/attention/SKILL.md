---
description: Set what hands says unprompted, or say what is set. Kinds and levels, given in pairs - finished (each turn a session finishes) off, brief, or full; progress (the focused session's steps as it works) off, brief, or full; ended (a session ending) on or off; quiet on or off, which holds all of them, whatever their level, until it is off again. Off, a thing is told only when asked for. Permission requests, questions asked in a dialog, and plans are always spoken. Takes effect from the next thing hands would have said, with no restart, and lasts across restarts.
argument-hint: "[finished|progress|ended|quiet LEVEL]..."
allowed-tools: Bash(*hands.sessions.attention*)
---

!`"${CLAUDE_PLUGIN_ROOT}/hooks/python" -m hands.sessions.attention $ARGUMENTS`

Tell the user, in one line, what the line above says. Do nothing else.
