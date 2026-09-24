# Features

The work that delivers the nine [needs](needs.md), organised as epics in lit. Each
ticket names the need it serves and the shape of done, so that "finished" has a test
before the work starts `[LAW:verifiable-goals]`. The order in `lit backlog` is the
order to build in; this document explains why that order and stops there, so the
two never disagree about rank.

## The ordering principle

Transport before content. In an LLM-in-the-loop system a content defect degrades
gracefully and a transport defect is a hard failure you feel on the first try
(failure mode 11). So the foundation epic, which delivers the spike's go/no-go and
the session, draft, and permission plumbing, comes first. **Narration** is the first
epic after it: hearing a spoken summary of each Claude turn is the most important
thing hands does, so the output path comes ahead of anything that types into a
session. **Loud daemon** follows, ahead of every feature that makes the intermediary
smarter. After that, the epics are ordered by how soon their absence
sends you back to the keyboard: interrupt and questions first, then attention across
sessions, then ending sessions, dictation, the phone, and the day-long memory.

Nothing in the plan is a flag. Where two behaviours are wanted, they are variants of
one config value with a declared cap `[LAW:no-mode-explosion]`; where one behaviour
is wanted for N things, it is one type and N values `[LAW:one-type-per-behavior]`.

## Foundation (existing epic `hands-architecture-3qr`)

The spike is closed: GO, with the latency numbers in the ticket. The reducer, the
draft buffer, voice permission approval and the own-voice bleed bug are closed with
it, which leaves the backfill reader. Built: `read_session` answers what a session
did before the daemon attached, reading its transcript through the same fold the
live tail runs, so a turn nobody heard and a turn heard live are told in the same
words. What was asked is kept in its place among what was done, because the steps
alone say how a session spent an hour and never what for. Everything names the record
it came from, and a reading hands over forty of them and where to read on from,
because an hour of work is hundreds and all of them at once is a context spent on
history — never a mark on a call the session has not come back from, whose result
would otherwise land after the mark and be told to nobody. Two refinements from the
architecture apply to them and are recorded as comments on the tickets: the reducer
and its types live in the `core` package with an import-boundary test, and the shim
writes the session file at `SessionStart`.

## Narration (`hands-narration-2mc`)

Need 4, and the precomputing half of need 7. How Claude's results reach the ear.
Nothing here reads text verbatim, and tool calls are content, not noise to filter.
The first eight tickets are the first working version; progress while working and
subagent narration follow it. A first slice of them runs today: each turn a session
finishes is summarised in one to three sentences and spoken with the session's name,
and a session's end is spoken after its last turn. The notes on the items below say
what each has and what remains.

- **Transcript tail and step recognisers.** While a session is registered, the
  adapter follows its JSONL from the watermark, and one table of recognisers turns
  records into typed `Step`s: text, edits with their patches, commands with their
  purpose, output, failure, and what they did to the repository, test runs with the
  failing names, reads and searches, task updates, `Agent` dispatches with their
  reports, and `AskUserQuestion`. An unrecognised tool is named and summarised,
  never dropped. Done by fixture tests on real JSONL slices, and by a measurement
  of the lag from a record's `timestamp` to its `Step`. Built: the recogniser table
  and its nine typed steps, fitted to shapes read out of 900 real transcripts and
  tested against fourteen real call-and-result pairs and the output four real test
  runners write; the tail that follows every live transcript from its watermark ten
  times a second, measured live at 96 to 305 ms from a record being written to its
  step, median 160 ms; and at `Stop`, the turn told from the tail rather than reread,
  skipping subagent records, with a turn that stops twice told only the steps the
  first stop did not tell. Remaining: steps reaching the reducer as events, which is
  what progress while working is built on.
- **The turn's git delta.** Built: at `UserPromptSubmit` the daemon records where the
  target's repository stands, without staging, stashing or reverting anything and
  without writing to the repository's own index; at `Stop` it reads the files changed,
  the commits made, and the diff, and the summariser is given them beside the steps.
  Files a turn created are in it, which is why the tree is written through a scratch
  index rather than a stash — a stash holds no untracked file, and a generated one is
  exactly what a turn must be able to name. Changes made by any means, a formatter or a
  `sed` in a shell command included, are part of the result, and a turn whose steps say
  nothing but whose repository moved is told rather than passed over in silence.
- **Spoken form.** A pure transform in `core`, installed as the TTS service's text
  transform so every utterance passes through it: headings become section cues,
  lists become counted sequences, identifiers are split into words, paths become
  file names, and flags, hashes, ids, and URLs are named or dropped. Code, diffs,
  and tables are summarised before they get here. Done by a table test of written
  inputs from real Claude replies and their spoken forms, and a test that no text
  reaching TTS contains a backtick, a pipe table, or a fenced block.
- **Summaries and the narration tree.** A stateless summariser call on the
  configured backend turns a turn's steps, final text, and git delta into a tree
  of segments: a headline, the questions, then one section per topic, each
  carrying its record ids and opening into children. Code and diffs are described
  by what they do. Step summaries are built as steps arrive, so the headline is
  ready at `Stop`. The top level's length is a config number that starts at one
  sentence and is expected to change as soon as it is heard. Done by an eval
  script over real turn fixtures that checks the steps' facts are present, no
  identifier or code is spoken, and the length holds, and by first audio after
  `Stop` measured on inferno. Built: a stateless summary of each finished turn, one to
  three sentences, spoken with the session's name, with a spoken sentence when it
  fails; first audio came 1.36 s after `Stop` on inferno. Remaining: the tree of
  segments, step summaries built as steps arrive, the git delta, the length as a config
  number, and the eval script.
- **Question detection.** Questions in the final text, and choices Claude offered,
  become question segments that play at every length, ahead of the sections; a
  turn that ends on one makes the idle nudge say the session has a question. Done
  by fixtures of real turns that do and do not end on a question, with the eval
  reporting misses and false alarms. Built: the summary instruction ends a turn's
  summary on the question it asked. Remaining: question segments, the idle nudge, and
  the fixtures and eval.
- **The intermediary's prompt.** The conversational model's own deliverable,
  separate from the summariser's: titles for sessions and no ids aloud, replies in
  spoken form, "speak what changed" for readbacks, calling `expand`, `resume`,
  `skip`, and `repeat` instead of paraphrasing from memory, and `stay_silent` for
  words not addressed to it. Done by an eval script
  over conversation fixtures, run against the local model.
- **Playback bookmarks and resume.** One player knows which segment is on the
  speaker. An interruption pushes a bookmark; "go back to what you were talking
  about" pops it and replays that segment from its start; "skip that" and "say
  that again" move the same cursor. Done by reducer table tests, and live: barge
  in mid-summary, ask something unrelated, and resume at the segment that was cut
  off.
- **Drill-down.** "More on that" expands the segment playing, or the last one
  played, into its children, built from its records on demand. There is no
  verbatim mode: the deepest level is a longer summary, and code is still
  described, not recited. Done live: a failing test named in a headline opens into
  what failed and why.
- **Progress while working.** Streaming narration: a focused session's steps from
  the tail, and its text line by line from `MessageDisplay`, play at `fyi` priority
  as they happen, coalesced so a burst of edits is one sentence, and a normal
  session's are notes. Done live: a focused session running tests is heard doing so
  before it stops, and a long explanation is summarised before it finishes.
- **Subagent narration.** After the first working version. A subagent's transcript
  under the session's `subagents/` directory is tailed like the parent's, its type
  and description come from the `.meta.json` beside it, and its report is
  summarised as its own narration linked to the parent's `Agent` call. Done when
  "what did the reviewer find" is answered from the subagent's own steps.

## Loud daemon (`hands-liveness-x20`)

Need 6, and the restart half of need 9. The first epic after narration. Its
tickets depend on nothing past the reducer, so it can start the moment that ticket
closes.

- **Daemon under launchd with a heartbeat.** `hands run` is the entry point; a
  launchd plist with `KeepAlive` supervises it; every heartbeat rewrites
  `~/.hands/status.json`; `hands status` prints it. Done when killing the daemon
  produces a restart within the launchd interval and `hands status` shows the new
  pid and uptime.
- **System speech channel.** The `Speak` effect becomes a `TTSSpeakFrame`; the
  daemon speaks its own start and restart, an unreachable LLM, an empty Whisper
  result, and a TTS error posts a macOS notification instead. Done when stopping
  `mlx_lm.server` on inferno is heard within one turn, with no model involved.
- **Membership from session files, liveness from pids.** The shim writes
  `~/.hands/sessions/<id>.json` at `SessionStart`; the daemon reads the directory at
  start and watches it; a liveness sweep moves dead pids to `Gone` and speaks it.
  Done when a daemon restart lists the same sessions it listed before, and closing a
  session's terminal is spoken within the sweep interval.
- **Audit log.** Every effect and every failure is one JSONL line; `hands log`
  tails it. Done when a dictation can be traced in the log from the transcribed
  words to the readback.
- **Screen path.** A hands menu-bar status item, run by its own launchd agent
  rather than the daemon, reads `status.json` and shows the verdict as its icon, and
  a macOS notification is posted when the daemon stops being up. Done when killing
  the daemon changes the icon and posts the notification within a few seconds.
- **Audio device loss.** Unplugging the headset does not kill the pipeline: the
  transport error is spoken through the surviving device or shown on screen, and
  the transport is rebuilt on the default device. Done by unplugging mid-turn.
- **Hook installer.** `hands install-hooks` merges the eight hook entries, with
  their timeouts, into the Claude Code settings file, idempotently, keyed by a
  marker so re-running changes nothing; `MessageDisplay` is an HTTP hook, the rest
  are shims. Done when running it twice yields one diff, and when a streaming turn
  with the daemon stopped stalls no longer than the hook's timeout.

## Whole keyboard by voice (`hands-keyboard-gxr`)

Need 1. The acts a keyboard performs that the foundation does not yet cover.

- **The `Input` union on the virtual keyboard.** Planned under
  `hands-harness-5nb`, whose design is open: how keys reach the right session's
  window, the macOS permission it needs, and confirming a send through the
  `UserPromptSubmit` hook. `Text` escapes a leading sigil, `Command` keeps it, `Key`
  sends a named chord; `send_command` and `interrupt_session` are the tools. Done
  when "/compact" reaches the target as a command, "slash compact" as text, and
  "stop it" sends Escape, each verified in a live session.
- **Questions through the permission hook.** `AskUserQuestion` arrives as a
  `PermissionRequest`; the `Blocked.on` becomes `Question`; the pipeline reads the
  options; `answer_question` replies with the answers in `updatedInput`. Done when a
  real `AskUserQuestion` in a target session is answered by voice and the agent
  proceeds with that answer. Built: the hook carries the answer, so no key chords.
- **Plan approval.** `ExitPlanMode` arrives the same way; the `Blocked.on` becomes
  `Plan`, and the model tells the plan at summary depth. `answer_plan` approves it
  back into the mode the session had before it planned, or into the mode the user
  names (edits auto-accepted, or each edit asked about), or keeps planning with
  what to change. Built: verified live on 2.1.281, each approval left plan mode for
  the mode meant, and a plan sent back stayed in plan mode with the feedback
  reaching the agent.
- **Mode readback.** `permission_mode` from every hook payload lands in
  `Session.mode`; `list_sessions` speaks it; a mode change is a `Note`. Done when
  shift-tab in a session is reflected in the next `list_sessions`.
- **Waiting-for-you nudge.** The `idle_prompt` notification becomes a `Speak`: "X
  is waiting for you." Done when leaving a session idle triggers exactly one nudge.
  Built: heard live on 2.1.280, once per idle period, 60 s after the turn ended.

## Attention (`hands-attention-ssy`)

Need 3. How several sessions share one ear.

- **Routing table and overlays.** The `DEFAULT_POLICY` table and the per-session
  overlay `focused | normal | muted`; `focus_session` and `mute_session` tools; every
  tool's `session` argument defaults to the focus. Done when muting a session turns
  its `Stop`s into notes and the model can still answer "what did it do" from them.
- **Priority queue with hold and coalesce.** Pending speech orders `blocking`
  before `result` before `fyi`, waits while the key is down, and a pure `coalesce`
  folds one session's pending items into one narration carrying all their record
  ids. Done by a table test on `coalesce` and a live test with two sessions
  finishing during one held key.
- **Transitions only.** Every announcement is keyed by the record id or request id
  that caused it and is never repeated; the deadline warning is the
  `warned: False → True` transition. Done by reducer tests: the same event twice
  yields one utterance.
- **Catch-up.** `catch_up(since)` reads the audit log and narrates what was said
  and done while you were away, at summary depth. Done when "what did I miss"
  after ten minutes lists every session that finished.
- **Quiet.** A global overlay under which only `blocking` speaks and everything else
  is a note. Done when "be quiet for a while" suppresses `Stop`s and a permission
  request still gets through.

## Ending a session by voice (`hands-lifecycle-n1m`)

Need 1, the lifecycle act. Sessions are started in a terminal, not by hands; a session
hands can type into is one the session-input epic (`hands-harness-5nb`) can reach.

- **End a session.** `end_session` types `/exit` as a `Command` into the session; the
  session ending moves it to `Gone` and it is spoken. Done live.

## Dictation fidelity (`hands-dictation-vpz`)

Need 2.

- **Vocabulary bias.** Whisper's `initial_prompt` is built from the focus
  session's identifiers: file basenames from `git ls-files`, branch names, and
  recent session titles. If the Pipecat MLX service does not expose the prompt, a
  subclass passes it. Done when "auth middleware" transcribes as `authMiddleware`
  in a repo that has that file, measured over ten utterances.
- **Path grounding.** `find_path(session, query)` returns matching paths from
  `git ls-files` in the target's `cwd`, names only; the draft records the
  resolution and the readback speaks it. Done when a spoken file reference lands
  in the staged draft as the real path.
- **Spelled identifiers.** "Spell it" and letter names produce the exact token in
  the draft and the readback confirms it letter by letter. Done by a fixture of
  spoken spellings and their expected tokens.

## Presence (`hands-presence-em5`)

Need 8. Where the microphone is.

- **Gate edges as a config variant.** `terminal | hotkey | button | web |
  wakeword` in `Config`; the `hotkey` edge is a macOS event tap that reports real
  key-down and key-up, including a headset's media key. Done when holding the
  hotkey in another app drives a turn and releasing ends it.
- **Phone over WebRTC.** `Transport.WebRTC` serves Pipecat's SmallWebRTC transport
  and one page with a hold-to-talk button; reachable over the LAN and Tailscale.
  Done when a full turn round-trips from a phone with earbuds and the transcript
  contains no words from the reply.
- **Wake word with VAD stop, half-duplex.** The `wakeword` edge opens the gate on
  the wake word and closes it when Silero reports the configured silence; the
  detector is deaf while the output transport plays. Done when a turn completes
  with no button and no own-voice words in the transcript, speakers on.
- **Acoustic echo cancellation.** The macOS voice-processing audio unit replaces
  PyAudio input, lifting half-duplex and closing the own-voice bleed bug. Done when
  barge-in with speakers on yields a clean transcript.

## Endurance (`hands-memory-5qk`)

Need 9, the memory half.

- **Context summarisation.** `LLMAutoContextSummarizationConfig` on the
  intermediary's context, with the recent turns kept verbatim. Done when a
  three-hour conversation stays under the configured token bound and the model
  still answers a question about its first ten minutes from the summary.
- **Recall.** `recall(query, since)` searches the audit log for what was said,
  drafted, and approved. Done when "what did we decide about the token helper"
  returns the draft that mentioned it.

## Config (`hands-config-60f`)

Cross-cutting; pulled when the first variant beyond the LLM backend arrives.

- **One file, parsed once.** `~/.config/hands/config.toml` parsed into a frozen
  `Config` in `daemon`; the spike's environment variables are deleted; the settings
  cap in `architecture.md` lists every field. Done when no module below `daemon`
  reads `os.environ` and a test proves it.
