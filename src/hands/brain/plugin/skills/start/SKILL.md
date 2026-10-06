---
name: start
description: How to start a new Claude Code session for the user. Load it whenever the user asks for a new session - in a project, on another model, or to try something out.
---

# Starting a session

`start_session` starts `claude` in a folder, in a new tmux window, as the user would start it at a terminal, and returns
once the session has joined hands.

- The folder is the project's repository. Find it with the shell, as you find a repository for its backlog, and ask the
  user only when you cannot.
- Give `model` only when the user named a model, as they named it: `opus`, `sonnet`, `haiku`, or a full model id.
  Without it, the session starts on Claude Code's own default.
- It returns the session's id and the tmux pane it runs in. The session is in your session listing from then on, named
  by its project like any other.

A started session has been told nothing. Starting one is not a send: when the user also said what it should do, stage
that for the new session, by its id, as any prompt is staged, and send it when they say to.

When it fails, it says why. A session that has not joined in time is usually at a dialog, such as Claude Code asking
whether to trust a folder it has never worked in; the error holds what its pane shows. Tell the user what it asks, and
answer it only as they say, in that pane, as you answer any session's dialog from the shell.

Say what you did in a sentence: "Started a session in billing, on Opus."

The user said: "start a session in billing on opus and have it fix the rounding bug"
WRONG: you start it and type "fix the rounding bug" into its pane, or you say "Started, and it's fixing the rounding bug."
RIGHT: you start it, stage "Fix the rounding bug." for the new session, and the readback asks the user to send it.
