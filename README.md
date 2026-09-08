# cc-hands

Hands-free control of Claude Code. You speak; an intermediary agent cleans up what you
said, sends it to the right session, watches what comes back, and tells you about it.
It's your hands when your own hands are otherwise occupied.

The name is deliberately specific. Every interface this consumes — hook payloads, the
`~/.claude/projects` transcript layout, `CLAUDE_CONFIG_DIR` — is Claude Code's. A
tool-agnostic rewrite would earn the name `hands`; this one hasn't.

## The relationship

You are the executive. The intermediary is not a delegate off doing its own thing and
not a deputy with standing authority to decide — it works hand in hand with you. It
holds your half-formed intent, improves it, confirms, and only then acts.

That distinction drives the whole design. **`send_draft` is the only call that writes to
a coding session**, and nothing calls it without your say-so.

## Why an intermediary at all

Dictation straight into a coding agent doesn't work. Speech-to-text mangles identifiers,
you revise yourself mid-sentence, and you leave out the context the agent needs. Sending
that raw to something holding an `Edit` tool is how you get confidently wrong work.

So the intermediary proofreads. That's the load-bearing use case, not a nicety. The
second one is follow-up: you hear a summary and ask "give me the details of that part."
That only works if the thing that summarized still holds what it summarized — which is
why summarization and Q&A are one stateful agent rather than a cheap stateless narrator.

## Architecture

Three processes.

```
lowtalker (STT) ──► daemon ──► tmux send-keys ──► controller session
                      ▲                                  │
                      │                            MCP (unix socket)
                      │                                  ▼
                      │                              daemon tools
                      │                                  │
                      │                          tmux send-keys
                      │                                  ▼
   hook shims ────────┘                        target Claude Code sessions
   (from every session)                         (any pane, started any way)

                    daemon ──► OpenAI TTS ──► speech queue ──► speakers
```

**The daemon** owns everything stateful: the session registry, the speech queue with a
single audio owner, the draft buffer, and the MCP server. One process, one unix socket.

**The controller** is a real interactive `claude` CLI session in its own tmux pane,
configured in isolation, restricted to the MCP tools and nothing else. Input arrives via
`tmux send-keys`. No SDK.

**The hook shims** are two-line scripts installed in the isolated config dir. Each POSTs
its stdin to the daemon socket and returns immediately.

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

`Stop` carries the turn's full text, so narration needs no transcript read.
`MessageDisplay` carries streaming `delta`s, so speech can start before a turn finishes.
`PermissionRequest` is a synchronous interception point — it's how voice approval works.

The documented nine events are not the real set. There are 33:

```
ConfigChange CwdChanged DirectoryAdded Elicitation ElicitationResult FileChanged
InstructionsLoaded MessageDisplay Notification PermissionDenied PermissionRequest
PostCompact PostModelSwitch PostToolBatch PostToolUse PostToolUseFailure PreCompact
PreModelSwitch PreToolUse SessionEnd SessionStart Setup Stop StopFailure SubagentStart
SubagentStop TaskCompleted TaskCreated TeammateIdle UserPromptExpansion
UserPromptSubmit WorktreeCreate WorktreeRemove
```

### Transcripts are for backfill only

When the controller attaches to a session that's been running for an hour, no hooks
fired for those turns. `read_session(since)` reads
`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl` to answer "what happened before I
got here." That is its only job.

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

## Controller isolation

The controller must not inherit the personal `~/.claude`. Left alone it would pull in
hooks that fire on every turn, a large skill catalog, five MCP servers, and a global
CLAUDE.md full of git-workflow and ticket-lifecycle mandates — none of which belong in a
session whose job is to say one sentence back to you.

```bash
CLAUDE_CONFIG_DIR=~/code/cc-hands/.controller \
claude \
  --system-prompt-file ./prompts/controller.md \
  --setting-sources '' \
  --mcp-config ./mcp.json --strict-mcp-config \
  --allowed-tools 'mcp__hands__*' \
  --session-id "$CTL_SESSION"
```

`--system-prompt` and `--system-prompt-file` do a **full replacement**, not an append.
`--setting-sources ''` blocks user, project, and local settings — which also closes the
trap that `CLAUDE_CONFIG_DIR` alone leaves open, where a project `CLAUDE.md` is picked up
from the working directory. Assigning `--session-id` means the controller's own
transcript path is known before the process starts.

Config lives in this repo, not in dotfiles. It's product configuration, versioned with
the thing it configures.

Flag *names* are verified against the installed bundle; argument arities are not. Check
`claude --help` before relying on the exact shapes above.

## Tool surface

```
list_sessions()
read_session(session, since?)
stage_draft(session, text)
amend_draft(session, text)
discard_draft(session)
send_draft(session)          # the only call that writes to a coding session
speak(text)
```

Nothing else. No Bash, no Read, no Edit — a controller that can edit files will
eventually decide to do the work itself, and that ends the experiment.

The draft buffer lives in the daemon rather than in the model's head. Models are
unreliable at holding "I am currently mid-draft" across a long conversation, and the
failure is expensive. Keeping it in the daemon makes the state inspectable, makes the
readback deterministic, and makes "did it send something I didn't approve" answerable
from one function's call log.

**Readback rule: speak what changed, not what you said.** "Sending: refactor the auth
middleware to use the new token helper — I read 'auth middleware' as
`authMiddleware.ts`" is worth three seconds. Reciting your own sentence back at you is
not, and you'll stop using it by day two.

## The constraint that will bite

Hooks run in the agent's critical path with a timeout — the bundle carries `timeoutMs`
and `budgetMs` per invocation, and `MessageDisplay` and `SessionStart` dispatch with
`forceSyncExecution: true`. A shim that blocks on TTS synthesis will stutter the agent's
own output.

Every shim POSTs and returns immediately. The one exception is `PermissionRequest`, where
blocking is the entire point — and there you're racing a timeout with your own voice, so
the daemon needs a default when you don't answer in time. Find out what that budget
actually is before depending on it.

## Turn-taking

Push-to-talk makes the mic key the turn boundary: unambiguous, no VAD, no crosstalk
heuristics, no wake word, no agent guessing whether you were talking to it. Barge-in is
a keybind that kills playback.

The one piece worth keeping from a realtime design is the output queue. When two sessions
finish at once, utterances line up behind a single audio owner instead of overlapping.

## Build order

1. `list_sessions`, `read_session`, `speak` — driven by **typing** at the controller.
   That's enough to answer "what's it doing" and "tell me more about that part," which is
   the half of the use case that proves the loop. If it's good typed, voice makes it
   better. If it's bad typed, voice won't rescue it.
2. Draft buffer and `send_draft`.
3. lowtalker on the input side.
4. `PermissionRequest` voice approval.

## Prior art

Happy (`~/code/happy`) does hands-free Claude Code over ElevenLabs ConvAI. Worth reading
for what to avoid: it dumps 50 raw records at connect **in reverse chronological order**,
never summarizes, streams every message change unbounded into one context window that is
never evicted, and reconstructs permission state from `agentState` diffs — which
double-announces and drops all but the first pending request. Its two good ideas are the
silent-context vs. speaks-now channel split, and the read watermark in its TTS path.
