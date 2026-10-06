---
name: close
description: How to close Claude Code sessions the user is finished with. Load it whenever the user asks to close, end, quit, or clear out a session, or the sessions that are done.
---

# Closing sessions

`close_session` ends a session: its claude exits as at a closed terminal, and it leaves your session listing. It returns
once it has. A tmux window hands opened for it closes with it; a terminal the user started it from stays open.

- A session the user named is closed with `asked` set to `named`, whatever it is doing: they said which one.
- For the sessions that are done, call it with `asked` set to `done` for every session listed in what they asked about:
  all of them, or those of the project they named. It ends only those at their prompt, with no dialog up and nothing
  running in the background, and leaves the rest running, saying what each is doing. Never pick the done ones out of
  list_sessions yourself: a session's state can change between your listing and the close, and the tool checks it as it
  closes.
- Close only what they asked about. Any session in your listing can be closed, whoever started it.

Say what you did in a sentence, by name: what you closed, and what you left running and why. When a close fails, it
says why; tell the user.

The user said: "close the ones that are done"
WRONG: you read list_sessions, pick out the idle ones, and close each with `named`.
RIGHT: you close every listed session with `done`, and say "Closed billing and docs; cc-hands is still working, so I
left it."
