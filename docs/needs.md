# What hands-free coding needs

A developer can work away from the keyboard and rely on cc-hands when nine conditions
are met. For each need below, this document describes what fails when the need is not
met, the part of the [architecture](architecture.md) that implements it, and the epic
in [features.md](features.md) that delivers it. Every feature exists to meet one or
more of these needs. A feature that does not meet any of them is not built.

The needs come from one exercise. We listed every action a developer performs at a
keyboard during a day of work with coding agents. For each action, we asked what is
required to perform it by voice, without looking at a screen and without returning to
the desk. If any action cannot be performed by voice, the developer has to return to
the keyboard, and the product has failed at that point. The requirement is therefore
not "voice is possible" but "the keyboard is never required."

## 1. Every keyboard act has a spoken equivalent

At the keyboard, you enter prompts, interrupt, answer questions, approve tools, accept
plans, select options, run `/compact`, and close sessions. Starting a session is not on
this list, because it happens in a terminal, where the user is already at the keyboard.
If any of these actions has no spoken equivalent, you must return to the desk the first
time you need it. The need is completeness, and it is verified by enumeration: the list
of actions is finite, and each action is either supported by voice or recorded as a
known gap.

Implemented by the `Input` union, the `Held` dialog with its three blockers, and the
tool surface. Delivered by the drafts and permissions tickets in the foundation epic,
followed by **Whole keyboard by voice** and **Ending a session by voice**.

## 2. Dictation lands as intended

Speech-to-text can turn "auth middleware" into a guessed filename and a flag into an
ordinary word. Perfect recognition is not possible, so the need is not perfect
recognition. The need is that the user hears how the input was interpreted and can
correct it before it is sent. The readback reports what changed. Identifiers are
biased toward the words that this repository uses. A spoken reference to a file
resolves to a real path. The user can spell out a difficult word.

Implemented by the draft buffer, which stores its resolutions, and by `find_path`.
Delivered by the drafts ticket and **Dictation fidelity**.

## 3. The right thing is said at the right time

Several sessions share one audio output. A permission request from one session must be
announced before a completed turn from another session. If three turns complete while
you are speaking, they are announced in one sentence. A request that is still pending is
not announced again. No announcement starts while you hold the key down. A muted
session produces no announcements. When nothing has changed, nothing is announced.
Announcements report changes in state, never the current state.

Implemented by the routing table, the per-session overlay, the priority queue, and the
`coalesce` pass. Delivered by **Attention**.

## 4. Claude's results are understandable by ear, at the depth you choose

The result of a turn is what Claude did, not only the text it wrote at the end. For
example: it changed the token refresh to retry three times, two tests in the auth suite
failed, it committed and opened a pull request, and it is asking whether to keep the
old endpoint. Most of this information is in tool calls and their results, and some of
it is only in the repository. The summary is therefore built from every source that
contains it: the text in the transcript, the tool calls and results in the transcript,
and the git changes made during the turn.

Nothing is read aloud verbatim. Markdown, code, diffs, paths, hashes, and tables cannot
be understood when read aloud as written, so all spoken output is first converted to
spoken form. Code is described by what it does, a path by its file name, and a table by
what it shows. A summary is read first, and each part of the summary can be expanded
into more detail on request. If Claude asked a question, the question is always read,
regardless of the summary's length. If you interrupt the reading to ask something, "go
back to what you were talking about" resumes the reading where it stopped. "That part"
refers to the records that the part was built from.

Implemented by the transcript tail, the step recognisers, the turn's git delta, the
spoken-form transform, the narration tree, and the playback bookmarks. Delivered by
**Narration**.

## 5. Nothing happens without you

The intermediary sends only what you approved, approves only what you approved, denies
requests that you do not answer, and can never edit a file. Every effect it performs is
recorded as one line in an audit log, so the question "did it send something I didn't
say" always has a definite answer. The tool surface is small and fixed. Tools can
access names and records only, never file contents.

Implemented by the draft buffer, the deny-by-default deadline, the audit log, and the
fixed tool list. Delivered across the foundation epic. The audit log is delivered in
**Loud daemon**.

## 6. A failure is never silent

In an audio system, a failure and a task that is still running produce the same output:
no sound. The user must be able to hear or see that the daemon is running. A hook that
cannot reach the daemon must fail visibly. If the model is unreachable, hands must
report this by voice. An error must reach you through a path that does not depend on
the component that failed.

Implemented by the system speech channel, the heartbeat file and the menu-bar indicator
that reads it, the terminal that the daemon runs in, and the shim's non-zero exit code
when the heartbeat shows that the daemon crashed, hung, or cannot be
read. Delivered by **Loud daemon**. This epic is ranked ahead of every content feature
because transport defects are the defects that users encounter on the first attempt
(failure mode 11).

## 7. Fast enough to feel like conversation

The measured spike is 1.4 s from key release to first audio on a plain turn and 4.3 s
when a tool call is involved. The need is a latency bound that holds under load, and
skipping the model wherever the model adds no value. A session completion is announced
with a template, not a summary. Steps are summarised from the transcript tail while the
turn is running, so the summary is ready when the turn ends. The prompt is cached.

Implemented by the `Speak` channel and by summaries built during the turn. Delivered
across all epics, with the measurement recorded in the latency observer.

## 8. It works where you are

At the desk, with a headset. Across the room, with a hotkey or a button. In another
room, on the phone, with the phone's earbuds and its own talk button. Eventually, with
no button at all. Hands-free activation is the last step, not the first, because an
open microphone in a room with speakers picks up the pipeline's own voice.

Implemented by the transport variant and the gate-edge variant. Delivered by
**Presence**.

## 9. It lasts a whole day

The intermediary's own context is the only part of the system that grows over time. It
must be compacted without losing the ability to answer "what did we decide this
morning," and a daemon restart must not lose the list of existing sessions. The audit
log is the long-term memory, the context window is the working memory, and the session
files record which sessions exist.

Implemented by the context summariser, `recall`, `catch_up`, and the session files.
Delivered by **Endurance** and by the restart work in **Loud daemon**.
