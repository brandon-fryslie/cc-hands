# Features

This document describes the work that delivers the nine [needs](needs.md), organised
as epics in lit. Each ticket names the need it serves and its completion criteria, so
that "finished" has a test before the work starts `[LAW:verifiable-goals]`. The order
in `lit backlog` is the build order. This document explains the reasons for that order
but does not restate the ranking, so the two cannot disagree about rank.

## The ordering principle

Transport work comes before content work. In an LLM-in-the-loop system, a content
defect degrades gracefully, but a transport defect is a hard failure that the user
notices on the first try (failure mode 11). For this reason, the foundation epic comes
first. It delivers the spike's go/no-go decision and the session, draft, and
permission plumbing. **Narration** is the first epic after it. Hearing a spoken
summary of each Claude turn is the most important thing hands does, so the output path
comes before any feature that types into a session. **Loud daemon** follows, ahead of
every feature that makes the intermediary smarter. The remaining epics are ordered by
how soon their absence forces the user back to the keyboard: interrupt and questions
first, then attention across sessions, then ending sessions, dictation, the phone, and
the day-long memory.

The plan contains no flags. Where two behaviours are needed, they are variants of one
config value with a declared cap `[LAW:no-mode-explosion]`. Where one behaviour is
needed for N things, it is implemented as one type with N values
`[LAW:one-type-per-behavior]`.

## Foundation (existing epic `hands-architecture-3qr`)

The spike is closed with a GO decision; the latency numbers are in the ticket. The
reducer, the draft buffer, voice permission approval, and the own-voice bleed bug are
also closed. The backfill reader is the remaining item. Built: `read_session` reports
what a session did before the daemon attached. It reads the session's transcript
through the same fold that the live tail uses, so a turn that was not heard live is
described in the same words as a turn that was. The user's prompts are kept in order
among the steps, because the steps alone show what a session did over an hour but not
why. Every item identifies the record it came from. Each read returns forty items
and the position to continue reading from. An hour of work produces hundreds of
items, and returning all of them at once would fill the context with history. The
continuation position is never placed on a tool call that has not yet returned;
otherwise that call's result would arrive after the position and never be reported.
Two refinements from the architecture apply to these tickets and are recorded as
comments on them: the reducer and its types are in the `core` package, with an
import-boundary test, and the shim writes the session file at `SessionStart`.

## Narration (`hands-narration-2mc`)

Need 4, and the precomputing half of need 7. This epic covers how Claude's results are
delivered as speech. Nothing in it reads text verbatim, and tool calls are treated as
content, not as noise to filter out. The first eight tickets make up the first working
version; progress while working and subagent narration come after it. Part of this
already works: each turn a session finishes is summarised in one to three sentences
and spoken with the session's name, and the end of a session is announced after its
last turn. The summary plays only when finished-turn narration is enabled
(`/hands:attention finished full`, or by telling hands). When it is off, which is the
default, a finished turn plays only the question it is waiting for the user to answer.
The notes on each item below describe what is built and what remains.

- **Transcript tail and step recognisers.** While a session is registered, the
  adapter reads its JSONL from the watermark onward, and a single table of
  recognisers converts records into typed `Step`s: text; edits with their patches;
  commands with their purpose, output, failure, and effect on the repository; test
  runs with the names of failing tests; reads and searches; task updates; `Agent`
  dispatches with their reports; and `AskUserQuestion`. An unrecognised tool is named
  and summarised, never dropped. Done when verified by fixture tests on real JSONL slices, and
  by a measurement of the lag from a record's `timestamp` to its `Step`. Built: the
  recogniser table and its nine typed steps, fitted to record shapes taken from 900
  real transcripts and tested against fourteen real call-and-result pairs and the
  output of four real test runners; the tail, which reads every live transcript from
  its watermark ten times a second, with a measured live latency of 96 to 305 ms
  (median 160 ms) from a record being written to its step; and at `Stop`, narration
  of the turn from the tail instead of rereading the transcript. If a turn stops
  twice, the second narration includes only the steps the first one did not cover.
  Remaining: delivering steps to the reducer as events, which progress while working
  depends on.
- **The turn's git delta.** Built: at `UserPromptSubmit`, the daemon records the
  current state of the target's repository. It does not stage, stash, or revert
  anything, and it does not write to the repository's own index. At `Stop`, it reads
  the changed files, the new commits, and the diff, and passes them to the summariser
  along with the steps. The delta includes files the turn created. For this reason the
  tree is written through a scratch index instead of a stash: a stash does not include
  untracked files, and a turn must be able to report generated files. Changes made by
  any means, including a formatter or a `sed` in a shell command, are part of the
  result. If a turn's steps show no changes but its repository changed, the change is
  reported instead of being omitted.
- **Spoken form.** A pure transform in `core`, installed as the TTS service's text
  transform so that every utterance passes through it. Headings become section cues,
  lists become counted sequences, identifiers are split into words, paths become file
  names, and flags, hashes, ids, and URLs are either named or dropped. Code, diffs,
  and tables are summarised before they reach this transform. Done when verified by a table test
  that maps written inputs from real Claude replies to their spoken forms, and by a
  test that confirms no text reaching TTS contains a backtick, a pipe table, or a
  fenced block.
- **Summaries and the narration tree.** A stateless summariser call on the
  configured backend converts a turn's steps, final text, and git delta into a tree
  of segments: a headline, the questions, and then one section per topic. Each
  section carries its record ids and expands into child segments. Code and diffs are
  described by what they do. Step summaries are built as steps arrive, so the
  headline is ready at `Stop`. The length of the top level is a config number. It
  starts at one sentence and is expected to change once it has been tested by ear.
  Done when verified by an eval script over real turn fixtures, which checks that the steps'
  facts are present, that no identifier or code is spoken, and that the length limit
  is met, and by measuring the time to first audio after `Stop` on inferno. Built: a
  stateless summary of each finished turn, one to three sentences long, spoken with
  the session's name; if summarising fails, a spoken sentence reports the failure.
  First audio played 1.36 s after `Stop` on inferno. Remaining: the tree of segments,
  step summaries built as steps arrive, the git delta, the length as a config number,
  and the eval script.
- **Question detection.** Questions in the final text, and choices that Claude
  offered, become question segments. These play at every length, ahead of the
  sections. Done when verified by fixtures of real turns that do and do not end with a
  question, with the eval reporting misses and false positives. Built: the daemon
  checks a turn's closing text and its unanswered `AskUserQuestion` calls to find what
  the turn is waiting on. The question segment plays last at every length. It uses
  the summariser's wording, or, if the summariser left the question out, Claude's
  wording converted to spoken form. Nine eval cases, four with a question and five
  without, run with no misses and no false positives.
- **The intermediary's prompt.** The prompt for the conversational model, separate
  from the summariser's prompt. It covers referring to sessions by title and never
  speaking ids, replying in spoken form, "speak what changed" for readbacks, calling
  `expand`, `resume`, `skip`, and `repeat` instead of paraphrasing from memory, and
  calling `stay_silent` for speech not addressed to it. Done when verified by an eval script
  over conversation fixtures, run against the local model.
- **Playback bookmarks and resume.** A single player tracks which segment is
  currently playing. An interruption pushes a bookmark. "Go back to what you were
  talking about" pops the bookmark and replays that segment from the beginning;
  "skip that" and "say that again" move the same cursor. Done when verified by reducer table
  tests, and live: barge in mid-summary, ask something unrelated, and confirm that
  playback resumes at the segment that was cut off.
- **Drill-down.** "More on that" expands the turn that hands last narrated for a
  session into its parts, one line each. Asking for a part by name expands it into
  its records. Each repeated request gives more detail, and the detail is generated
  on demand. There is no verbatim mode: the deepest level is a longer summary, and
  code is still described, not read out. Done when shown live: when a headline names a
  failing test, expanding it reports what failed and why.
- **Progress while working.** Streaming narration. For the focused session, steps
  from the tail and text from `MessageDisplay`, line by line, play at `fyi` priority
  as they happen. They are coalesced so that a burst of edits produces one sentence.
  For a normal session, these updates are notes. Done when shown live: hands reports that a
  focused session is running tests before the session stops, and summarises a long
  explanation before it finishes.
- **Subagent narration.** Planned after the first working version. A subagent's
  transcript, in the session's `subagents/` directory, is read with the parent's fold
  at the point where the subagent reports back. The subagent is named by the task its
  parent gave it, and its steps form an expandable part of that turn. The link is the
  `agentId` that both the parent's call and the notification contain. Done when hands
  answers "what did the reviewer find" from the subagent's own steps.

## Loud daemon (`hands-liveness-x20`)

Need 6, and the restart half of need 9. This is the first epic after narration. Its
tickets depend only on the reducer, so work can start as soon as that ticket closes.

- **Daemon in a terminal with a heartbeat.** `hands run` is the entry point and
  runs in the foreground of a terminal. Every heartbeat rewrites
  `~/.hands/status.json`, and `hands status` prints it. Done when `hands status`
  reports the daemon as down after it is killed, and shows the new pid and uptime
  after it is started again.
- **System speech channel.** The `Speak` effect becomes a `TTSSpeakFrame`. The
  daemon announces its own start and restart, an unreachable LLM, and an empty
  Whisper result. A TTS error posts a macOS notification instead. Done when an
  unreachable model server is announced within one turn, without involving a model.
- **Membership from session files, liveness from pids.** The shim writes
  `~/.hands/sessions/<id>.json` at `SessionStart`. The daemon reads the directory at
  startup and watches it. A liveness sweep moves sessions with dead pids to `Gone` and
  announces the change. Done when a restarted daemon lists the same sessions as
  before, and closing a session's terminal is announced within the sweep interval.
- **Audit log.** Every effect and every failure is written as one JSONL line;
  `hands log` tails the log. Done when a dictation can be traced in the log from the
  transcribed words to the readback.
- **Screen path.** A hands menu-bar status item reads `status.json` and shows the
  daemon's status as its icon. It is a separate process that `hands run` starts, and
  it outlives the daemon long enough to announce that the daemon stopped. A macOS
  notification is posted when the daemon is no longer up. Done when killing the
  daemon changes the icon and posts the notification within a few seconds.
- **Audio device loss.** Unplugging the headset does not stop the pipeline. The
  transport error is spoken through the remaining device or shown on screen, and the
  transport is rebuilt on the default device. Done when verified by unplugging the headset
  mid-turn.
- **Hooks as a plugin.** The repository is a Claude Code marketplace. Its plugin is
  written by the installed hands (`hands plugin`), its hooks run under that
  installation's own Python, and its `hooks/hooks.json` is generated from
  `hookconfig`. Installing, disabling, or uninstalling the plugin turns the hooks on
  and off. If a shim cannot reach the daemon, it exits 0 without output when the
  heartbeat shows that hands was stopped or never ran. It exits 1 and reports the
  reason when the daemon died or hung, or when the heartbeat cannot be read. Done when
  a session with the plugin and no daemon shows no hook error and shows its own
  permission dialog, and a daemon killed with -9 causes a hook error.

## Whole keyboard by voice (`hands-keyboard-gxr`)

Need 1. Covers the keyboard actions that the foundation does not yet support.

- **The `Input` union, through fritter.** `Text` escapes a leading sigil, `Command`
  keeps it, and `Key` sends a named key chord. The tools are `send_command` and
  `interrupt_session`. A command is rejected while a dialog is open, but an interrupt
  still sends Escape. Built: on 2.1.283, "/compact" reached a live session as a
  command, "slash compact" reached it as text, and "stop it" sent Escape, which ended
  a running turn and closed a question dialog.
- **Questions through the permission hook.** `AskUserQuestion` arrives as a
  `PermissionRequest`. The `Held.on` becomes `Question`, the pipeline reads the
  options, and `answer_question` replies with the answers in `updatedInput`. Done
  when a real `AskUserQuestion` in a target session is answered by voice and the
  agent proceeds with that answer. Built: the hook delivers the answer, so no key
  chords are sent.
- **Plan approval.** `ExitPlanMode` arrives the same way. The `Held.on` becomes
  `Plan`, and the model reads a summary of the plan aloud. `answer_plan` either
  approves the plan and returns the session to the mode it was in before planning,
  approves it into a mode the user names (edits auto-accepted, or approval required
  for each edit), or keeps the session in planning with the requested changes. Built:
  verified live on 2.1.281. Each approval left plan mode for the intended mode, and a
  plan sent back stayed in plan mode, with the feedback reaching the agent.
- **Mode readback.** `permission_mode` from every hook payload is stored in
  `Session.mode`. `list_sessions` reports it, and the brain's request tail includes
  it. Done when a shift-tab mode change in a session appears in the next
  `list_sessions`.
- **One turn summary.** Every finished turn is summarised once, and the user hears
  only that summary of it. The summary plays when the turn finishes if finished-turn
  narration is on (brief or full), or if the session is watched and finished-turn
  narration is off. Otherwise it plays when the user asks for it (`tell_turn`). A
  muted session's summary, or any summary while hands is quiet, plays only when the
  user asks. `set_overlay` sets a session to watched, normal, or muted. Both settings
  are set by voice and persist across restarts. Nothing is announced for a session
  that is idle at its prompt (hands-narration-2mc.d52).
- **An interrupted turn.** Pressing Escape or Ctrl-C mid-turn fires no `Stop` and no
  later `idle_prompt`. Instead, the tail reads the interrupt record in the
  transcript, which identifies the prompt of the interrupted turn. Only that turn is
  set to idle. It is narrated in the same way as a stopped turn, beginning with "You
  interrupted it.". Built: verified live on 2.1.281 during a tool call, during text
  output, at a permission dialog, and with Ctrl-C. In each case the turn was set to
  idle and narrated.

## Attention (`hands-attention-ssy`)

Need 3. Covers how narration from several sessions is delivered to one listener.

- **Routing table and overlays.** The per-session overlay `normal | watched | muted`,
  set by `set_overlay`, and `delivery`, the table that combines the overlay with the
  narration settings. A muted session's `Stop`s are held and not spoken until the
  user asks for one (`tell_turn`), but its questions are still spoken.
  `focus_session`, and defaulting every tool's `session` argument to the focused
  session, are covered by hands-attention-ssy.xfs. Built: tests/test_attention.py;
  not verified live.
- **Priority queue with hold and coalesce.** The pending speech queue orders
  `blocking` before `result` before `fyi`, and waits while the key is held down. A
  pure `coalesce` function merges one session's pending items into a single narration
  that carries all their record ids. Done when verified by a table test on `coalesce` and a
  live test in which two sessions finish during one key hold.
- **Transitions only.** Every announcement is keyed by the record id or request id
  that triggered it and is never repeated. The deadline warning is triggered by the
  `warned: False → True` transition. Done when verified by reducer tests: the same event
  received twice produces one utterance.
- **Catch-up.** `catch_up(minutes)` reads the audit log and gives a spoken summary of
  what was said and done while the user was away. Done when "what did I miss", asked
  after ten minutes, lists every session that finished and the events that sessions'
  hooks reported, whether or not hands announced them.
- **What hands says unprompted.** A single control, `attention`, covers finished
  turns, the focused session's progress, and session endings. Each can be on or off
  and has a level. The control also has a quiet setting, in which only blocking items
  are spoken and everything else waits until the user asks for it. The setting is
  changed with one short utterance, takes effect from the next thing hands would say,
  and persists across restarts. Built: tests/test_attention.py,
  tests/test_progress.py, tests/test_narrator.py; not verified live.

## Ending a session by voice (`hands-lifecycle-n1m`)

Need 1, the session lifecycle action. Sessions are started in a terminal, not by
hands. hands can type only into sessions that the session-input epic
(`hands-harness-5nb`) can reach.

- **End a session.** `end_session` types `/exit` as a `Command` into the session.
  When the session ends, it moves to `Gone` and the change is announced. Verified
  live.

## Dictation fidelity (`hands-dictation-vpz`)

Need 2.

- **Vocabulary bias.** Each hold is transcribed with a Whisper `initial_prompt` that
  is built at transcription time (`hands.voice.vocabulary`). It contains the names of
  the files changed in the focused session's last 30 commits and in its uncommitted
  work, then its branch, then the project and name of every running session, newest
  last. It holds at most 40 of these, and the oldest are dropped until the rest fit in
  the 223 prompt tokens that Whisper keeps. For each hold, the prompt words and their
  token count are written to the audit log as a `Primed` line. Whisper's output is
  written as a `HoldHeard` line, which includes each segment dropped as not spoken and
  the hold's volume before and after the echo canceller. With a prompt, Whisper turns
  noise into "." or a guess ("and slow-talking.", average log probability -2.8) or a
  loop (compression ratio 17); these segments are dropped. Measured on 2026-10-03 with
  that code, using ten sentences spoken by `say` in a repository with
  `authMiddleware.ts` among thirty files: `authMiddleware` was recognised 8/10 times
  with the prompt and 0/10 times without it. Both misses were the voice saying "middle
  way". Built: tests/test_vocabulary.py; not verified live.
- **Path grounding.** `find_path(session, query)` returns matching paths from
  `git ls-files` in the target's `cwd` (names only). The draft records the resolved
  path, and the readback speaks it. Done when a spoken file reference appears in the
  staged draft as the real path.
- **Spelled identifiers.** "Spell it" and letter names produce the exact token in
  the draft, and the readback confirms it letter by letter. Done when verified by a fixture of
  spoken spellings and their expected tokens.

## Presence (`hands-presence-em5`)

Need 8. Covers where the microphone is.

- **Gate edges as values of one type.** `held key | button | phone button | wake word`
  are values of `Edge` (`hands.voice.trigger`). The user switches the desk's trigger
  by voice. The `held key` edge is a macOS event tap that reports actual key-down and
  key-up events. Done when holding the key in another app starts a turn and releasing
  it ends the turn.
- **Engaged conversation.** One hold of Right Shift starts engaged mode. From then
  on, the user's voice starts each turn and end-of-turn detection ends it. Silero
  detects where speech starts and stops on the echo-cancelled desk microphone, and
  Smart Turn v3 decides whether a stop ends the turn or is a pause within it. Another
  hold ends engaged mode. Both changes play an audio cue. Done when one press is
  followed by three turns without touching a key, a pause mid-sentence does not send
  a partial sentence, and conversation in the room after disengaging does not start a
  turn.
- **Wake word.** Saying "Hey Jarvis", or the wake word set in config.toml, starts a
  turn without a key press, and end-of-turn detection ends it as in an engaged
  conversation. openWakeWord's pretrained model, or a model the user trained, listens
  to the echo-cancelled desk microphone and receives silence while hands is speaking.
  Done when, with speakers on, a turn completes without a key press and its
  transcript contains none of hands' own words, and conversation in the room without
  the wake word does not start a turn.
- **Phone over WebRTC.** A hold-to-talk page served over HTTPS on the LAN and the
  tailnet, available alongside the desk for the whole run. hands listens and speaks
  on whichever device started the last turn. Done when a full turn completes a round
  trip from a phone with earbuds and the transcript contains no words from the reply.
- **Wake word with VAD stop, half-duplex.** The `wake word` edge opens the gate on
  the wake word and closes it when Silero reports the configured period of silence.
  The detector ignores input while the output transport is playing. Done when a turn
  completes without a button and with no own-voice words in the transcript, with
  speakers on.
- **Acoustic echo cancellation.** WebRTC's AEC3 removes the speaker echo from every
  desk microphone buffer. This removes the need for the half-duplex mute and fixes
  the own-voice bleed bug. Done when barge-in with speakers on produces a clean
  transcript.

## Endurance (`hands-memory-5qk`)

Need 9, the memory half.

- **Context summarisation.** `LLMAutoContextSummarizationConfig` on the
  intermediary's context, with the recent turns kept verbatim. Done when a
  three-hour conversation stays under the configured token limit and the model can
  still answer a question about its first ten minutes from the summary.
- **Recall.** `recall(query, since)` searches the audit log for what was said,
  drafted, and approved. Done when "what did we decide about the token helper"
  returns the draft that mentioned it.

## Config (`hands-config-60f`)

Cross-cutting. Scheduled when the first variant other than the LLM backend is needed.

- **One file, parsed once.** `config.toml` in the home is parsed into a frozen
  `Config` in `daemon`. The spike's environment variables are removed. The settings
  cap in `architecture.md` lists every field. Done when only the modules that a
  process starts in read `os.environ`, and a test verifies this.
