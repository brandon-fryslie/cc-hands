# cc-hands

Hands-free control of Claude Code. You speak; an intermediary agent cleans up what you
said, sends it to the right session, watches what comes back, and tells you about it.
It's your hands when your own hands are otherwise occupied.

## The intermediary

The intermediary makes the workflow eyes-free as well as hands-free: a full
voice-to-voice loop that needs neither hands nor eyes. The workflow has been tested and
works well in practice.

A basic hands-free workflow involves proofreading STT output and re-reading the original
when TTS output is garbled by special characters. The intermediary handles both. To
proofread a message before it goes to Claude, have the agent read it back. Claude's
output becomes an interactive surface you explore on demand. The intermediary is
designed to be fast.

Follow-up works the same way: you hear a summary and ask "give me the details of that
part." Summarization and Q&A are one stateful agent, so the thing that summarized still
holds what it summarized.

Starting functionality:

- Take audio, write the prompt for the coding session
- Read a draft back before it is sent; amend, discard, or send it (`stage_draft`,
  `amend_draft`, `discard_draft`, `send_draft`)
- List the running sessions and read what a session has done (`list_sessions`,
  `read_session`)
- Speak what came back
- Answer follow-up questions about anything it summarized

## Architecture

Two processes.

```
mic ──► push-to-talk gate ──► Whisper (MLX) ──► Claude (API) ──► pocket-tts ──► speakers
                                                  │    ▲
                                       tool calls │    │ hook events, as frames
                                                  ▼    │
                                           sessions module
                                    registry · drafts · JSONL reader
                                                  │
                                           tmux send-keys
                                                  ▼
                                    target Claude Code sessions
                                    (any pane, started any way)
                                                  │
                                 hook shims ──► unix socket ──► sessions module
```

**The daemon** is one Python process, launchd-supervised, named `hands`. It holds two
modules with a one-way dependency: `voice` depends on `sessions`, never the reverse.

`sessions` is the Claude Code side: the hook socket the shims POST to, the session
registry, the JSONL backfill reader, the draft buffer, `tmux send-keys` to target
sessions, and permission replies. Its core is a pure reducer — state plus event in, new
state plus a list of effect descriptions out — and thin adapters perform the effects:
send keys, reply to a blocked shim, append an audit record. The core describes effects;
it never performs them.

`voice` is the Pipecat pipeline: microphone, push-to-talk gate, Whisper on MLX, Claude
over the Anthropic API, pocket-tts, speakers. The LLM's tools are Python functions
registered on Pipecat's LLM service that call into `sessions`. Hook events enter the
pipeline as frames.

**The hook shims** are two-line scripts in every target Claude Code session. Each POSTs
its stdin to the daemon socket and returns immediately. `PermissionRequest` is the one
that blocks.

### Stack

Pipecat ships every piece the pipeline needs: an in-process pocket-tts service, a local
audio transport over PyAudio, a Whisper service with an MLX build
(`pipecat-ai[mlx-whisper]`, models such as `mlx-community/whisper-large-v3-turbo`), an
Anthropic LLM service, and function registration on that service. Verified from the
upstream repos on 2026-09-11.

pocket-tts is MIT, 100M parameters, CPU-only by design, and streams: about 200 ms to
the first audio chunk and about 6x real time on an M4 MacBook Air CPU. It clones a voice
from a wav and exports it to safetensors for fast load. Kyutai measured no GPU speedup
on Apple silicon — the model is tiny and batch size is 1 — so PyTorch's Mac GPU story is
irrelevant here: it runs on two CPU cores with the CPU-only wheels macOS gets by default.
If CPU load ever matters, a community MLX port (`pocket-tts-mlx`) is reachable behind a
small custom Pipecat TTS service. That is an escape hatch, not the plan.

STT is Whisper on MLX: Apple-silicon native, no torch in that path.

The LLM model is chosen by measured latency in the first spike.

Python, `uv`, pyright strict. State types are discriminated unions: dataclasses with a
`Literal` kind field.

### Hooks are the event source, not the transcripts

Hooks are lower latency than tailing JSONL and carry more data. Verified payload fields
(read out of the 2.1.263 bundle — field names are from the code, not from a live run):

| Event | Payload fields |
|---|---|
| `Stop` | `stop_hook_active`, `last_assistant_message` |
| `MessageDisplay` | `turn_id`, `message_id`, `index`, `final`, `delta` |
| `PermissionRequest` | `tool_name`, `tool_input`, `permission_suggestions` |
| `SubagentStop` | `agent_id`, `agent_transcript_path`, `agent_type`, `last_assistant_message` |
| `Notification` | `message`, `title`, `notification_type` |
| `UserPromptSubmit` | `prompt`, `session_title` |
| `SessionStart` | `source`, `agent_type`, `model`, `session_title` |

`Stop` carries the turn's full text but no record id, so the daemon does one tail read
of the session JSONL at each `Stop` to attach the `uuid`. `MessageDisplay` carries
streaming `delta`s, so speech can start before a turn finishes. `PermissionRequest` is a
synchronous interception point — it's how voice approval works.

The documented nine events are not the real set. There are 33:

```
ConfigChange CwdChanged DirectoryAdded Elicitation ElicitationResult FileChanged
InstructionsLoaded MessageDisplay Notification PermissionDenied PermissionRequest
PostCompact PostModelSwitch PostToolBatch PostToolUse PostToolUseFailure PreCompact
PreModelSwitch PreToolUse SessionEnd SessionStart Setup Stop StopFailure SubagentStart
SubagentStop TaskCompleted TaskCreated TeammateIdle UserPromptExpansion
UserPromptSubmit WorktreeCreate WorktreeRemove
```

### Push pointers, pull content

The LLM's context window holds **the conversation with you** — not session transcripts.
Hook events are injected as small frames carrying the session title, the event kind, the
record's `uuid`, and for `Stop` the turn's final text, which is already in the payload.
Anything older is a `read_session` query. Nobody asks about a message from thirty turns
ago, and when they do, `read_session` fetches it.

Which events run the LLM: `Stop` and `PermissionRequest` append with `run_llm` on.
`SubagentStop` is a deliberate separate choice. Everything else is either silent context
or not delivered at all.

Pipecat's `LLMMessagesAppendFrame` has a `run_llm` flag. Appending with it off is
Happy's silent-context channel; appending with it on is the speaks-now channel. That is
the same distinction Happy got right (see
[docs/happy-voice-reference.md](docs/happy-voice-reference.md)).

This is the single decision that avoids most of Happy's trouble. Happy pushed history in
and had no way to pull, so it needed a bootstrap dump, an eviction policy it never wrote,
and a window that only grew. Remove the push and all three problems stop existing.

The one thing that must be preserved for "give me the details of that part" to resolve:
**every narration carries the `uuid` of the record it came from.** Then "that part" is a
lookup rather than a fuzzy search back through what it said. Cheap at the source,
impossible to retrofit.

### Transcripts are for backfill only

When the daemon comes up after a session has been running for an hour, no hooks fired
for those turns. `read_session(since)` reads
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl` to answer "what happened before I
got here." That, plus the one-record tail read at `Stop`, is all the transcripts are
read for.

Record shapes worth knowing:

- `ai-title` → `aiTitle`. Live session name, free, no model call. Use it for
  `list_sessions` labels.
- `assistant` → `.message.content[]`, blocks typed `text` | `thinking` | `tool_use`.
  Narrate `text` only.
- `user` → `.message.content` is a plain string for real user turns, or an array of
  `tool_result`; `isMeta: true` marks injected reminders. Filter to string + non-meta.
- `mode` / `permission-mode` → live session mode.
- Every record: `uuid` (watermark key), `parentUuid` (turn tree), `timestamp`, `cwd`,
  `gitBranch`, `sessionId`.

Two filters are non-negotiable: `isSidechain: false`, or you narrate every subagent's
internal chatter; and the block-type filter, because in a sampled real session only
**6 of 69 assistant content blocks were `text`** — the rest were `thinking` and
`tool_use`. Budget speakable blocks, never records.

### Liveness

Silence says nothing about whether a session is alive. A session waiting for input is
silent for hours and alive; a session in a tool loop is never silent. So the shim
reports its parent pid at `SessionStart`, and liveness is a process check: one OS fact
instead of a timestamp heuristic.

## Tool surface

```
list_sessions()
read_session(session, since?)
stage_draft(session, text)
amend_draft(session, text)
discard_draft(session)
send_draft(session)
answer_permission(request, decision)
```

Nothing else. The intermediary cannot read or edit files; it stages text and sends it.

The draft buffer lives in the daemon rather than in the model's head. Models are
unreliable at holding "I am currently mid-draft" across a long conversation, and the
failure is expensive. Keeping it in the daemon makes the state inspectable, makes the
readback deterministic, and makes "did it send something I didn't approve" answerable
from one function's call log.

**Readback rule: speak what changed, not what you said.** "Sending: refactor the auth
middleware to use the new token helper — I read 'auth middleware' as
`authMiddleware.ts`" is worth three seconds. Reciting your own sentence back at you is
not, and you'll stop using it by day two.

### Sending while a target is busy

Text beginning with `/` or `@` triggers Claude Code's own completion, so the daemon
escapes leading sigils. Sending mid-turn races the input box, and whether
`tmux send-keys` mid-turn lands in Claude Code's own input queue is unverified. The
spike decides between sending immediately and `send_draft` returning a typed
refused-busy result. The daemon never holds a hidden queue.

## The constraint that will bite

Hooks run in the agent's critical path with a timeout — the bundle carries `timeoutMs`
and `budgetMs` per invocation, and `MessageDisplay` and `SessionStart` dispatch with
`forceSyncExecution: true`. A shim that blocks on TTS synthesis will stutter the agent's
own output.

Every shim POSTs and returns immediately. The one exception is `PermissionRequest`, where
blocking is the entire point — and there you're racing a timeout with your own voice.
The budget is ours to set: each hook entry in Claude Code settings takes a `timeout`
field, so the shim's hook config declares it and the daemon's default-deny deadline is
derived from the same number.

## Permission approval

The `PermissionRequest` shim blocks. The daemon injects the request with `run_llm` on,
the pipeline speaks it, you answer by voice, the LLM calls `answer_permission`, and the
daemon replies to the blocked shim with `{"behavior":"allow"}` or
`{"behavior":"deny","message":"..."}`. That reply shape is verified from the Claude Code
2.1.263 binary. If you haven't answered by the declared timeout, the daemon denies.

## Turn-taking

Push-to-talk makes the mic key the turn boundary: unambiguous, no VAD, no crosstalk
heuristics, no wake word, no agent guessing whether you were talking to it. A hotkey
processor in the pipeline emits Pipecat's user-started-speaking and user-stopped-speaking
frames and gates the microphone audio; VAD is off.

Pressing the key while the pipeline is speaking emits an interruption, which flushes
queued audio: that is barge-in. Because the mic is closed unless the key is held, TTS
output cannot leak into the mic.

Pipecat's output transport is the single audio owner. When two sessions finish at once,
their utterances line up behind it instead of overlapping.

## Build order

1. Pipeline spike: mic, push-to-talk gate, Whisper on MLX, Claude over the API,
   pocket-tts, speakers, and one stub `list_sessions` tool. Measure voice-to-voice
   latency and whether push-to-talk is clean. This is the go/no-go for the whole
   architecture.
2. Session state types and the pure reducer, with tests for every lifecycle transition;
   then hook shims, the registry, and a real `list_sessions`.
3. `read_session` backfill from JSONL with the two non-negotiable filters.
4. Draft buffer, `send_draft`, `tmux send-keys`, and the busy-target decision.
5. Voice permission approval.

The spike is step 1 for the reason failure mode 11 gives: in an LLM-in-the-loop system,
content defects degrade gracefully and transport defects are the ones you feel on the
first try. So transport is tested first and hardest.

## Prior art

Full writeup in [docs/happy-voice-reference.md](docs/happy-voice-reference.md), and the
catalogue of what goes wrong — observed and anticipated, each with the rule that prevents
it — in [docs/failure-modes.md](docs/failure-modes.md).

Happy (`~/code/happy`) does hands-free Claude Code over ElevenLabs ConvAI. Worth reading
for what to avoid: it dumps 50 raw records at connect **in reverse chronological order**,
never summarizes, streams every message change unbounded into one context window that is
never evicted, and reconstructs permission state from `agentState` diffs — which
double-announces and drops all but the first pending request. Its two good ideas are the
silent-context vs. speaks-now channel split, and the read watermark in its TTS path.
