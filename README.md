# cc-hands

Hands-free control of Claude Code. You speak; an intermediary agent cleans up what you
said, types it into the right session, watches what comes back, and tells you about it.
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

What it does, once built:

- Takes audio and writes the prompt for the coding session; reads a draft back until it
  is right, amends or discards it, and sends it by typing it into the session with a
  virtual keyboard, which is planned.
- Lists the running sessions, reads what a session has done, starts and ends sessions,
  and switches which one you are talking to.
- Answers permission prompts, questions, and plan approvals by voice, and denies by
  default when you don't answer.
- Interrupts a session and runs its slash commands.
- Speaks what came back at summary depth, with the details, the exact text, or the
  test output on request, and answers follow-up questions about anything it said.

## Where the design lives

- [docs/needs.md](docs/needs.md): the nine things that must be true before a developer
  can leave the keyboard for a working session.
- [docs/architecture.md](docs/architecture.md): the daemon, its four packages, the
  types at every seam, the three speech channels, how hooks and transcripts are used,
  and how failure stays loud.
- [docs/features.md](docs/features.md): the epics that deliver the needs, each with the
  shape of done; `lit backlog` holds the build order.
- [docs/failure-modes.md](docs/failure-modes.md): what goes wrong, observed and
  anticipated, and the rule each entry produced.
- [docs/happy-voice-reference.md](docs/happy-voice-reference.md): how Happy, the
  closest prior art, works, and what to copy and avoid.

## Shape

One launchd-supervised Python process, `hands`, with a Pipecat voice pipeline on one
side and the Claude Code plumbing on the other, meeting through a pure core.

```
mic ──► gate ──► Whisper (MLX) ──► LLM ──► pocket-tts ──► speakers
                                    │   ▲
                         tool calls │   │ hook events, as frames
                                    ▼   │
                             sessions + core
                     registry · drafts · JSONL reader · audit log
                                    │
                     hook replies · virtual keyboard (planned)
                                    ▼
                     target Claude Code sessions (any terminal, started any way)
                                    │
                  hook shims ──► unix socket ──► sessions
```

The LLM is a backend variant: `HANDS_LLM=local`, the default, is Qwen3-30B-A3B on
inferno through `mlx_lm.server`; `HANDS_LLM=openai` is `gpt-4.1-mini` through OpenAI's
API; `HANDS_LLM=anthropic` is Claude through the API. `HANDS_LLM_MODEL` names another
model for any of them. `HANDS_LLM_URL` moves `local` or `openai` to another
OpenAI-compatible server; it is the base URL the client appends `/chat/completions` to,
so it usually ends in `/v1` (`https://api-chicago.codexapi.pro/v1`: the bare host answers
404). Such a server must stream tool calls, because the pipeline's service always
streams: api-chicago.codexapi.pro streams replies but drops tool calls (2026-09-25). A keyed variant stops at start, naming the variable, when its key
(`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) is not set. The key can live in a `.env` at the
repository root, which git ignores, and `uv run --env-file .env` puts it in the
environment; uv stops if the file is not there. The
gate is push-to-talk: the key is the voice activity detector and the microphone mute,
so the turn boundary is the key and the pipeline can never transcribe itself. Measured
on 2026-09-12, voice to voice with the local model: 1.4 s from key release to first
audio on a plain turn, 4.3 s on a turn with a tool call.

## Running

```
uv sync
uv run hands run                        # in a terminal: space to talk, space again to stop, q to quit
HANDS_LLM=anthropic ANTHROPIC_API_KEY=... uv run hands run
HANDS_LLM=openai uv run --env-file .env hands run    # OPENAI_API_KEY=... in .env
uv run hands status                     # up, stopped, not responding, down, or never ran; exits 0 only when up
uv run hands log                        # the audit log: what hands heard, said, called, and failed at
uv run hands indicator                  # the daemon's verdict in the menu bar; launchd runs it this way
uv run hands install-hooks              # merge hands' hooks into ~/.claude/settings.json; again changes nothing
uv run pytest && uv run pyright
uv run python evals/narration.py       # real turns through the real summariser; needs the model to be up
```

`pytest` and `pyright` judge the code. The eval judges what a listener hears: it tells four
real turns, lifted whole out of real transcripts, and checks that the facts are there, that
no code name reached the ear, that what is spoken stays within its configured length, and
that every number the model said is a number the turn showed. It exits 0 when every check held, 1 when one
failed, and 2 when the model could not be reached at all.

launchd keeps the daemon up, starting it at login and again whenever it exits. The
menu-bar indicator gets an agent of its own, so it can still show the daemon when the
daemon is down:

```
uv run hands launchd daemon > ~/Library/LaunchAgents/hands.daemon.plist
uv run hands launchd indicator > ~/Library/LaunchAgents/hands.indicator.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/hands.daemon.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/hands.indicator.plist
```

The indicator's title is ✋ while the daemon is up. It reads "hands stuck", "hands down",
or "hands unreadable" when something is wrong, and "✋ off" when the daemon was stopped
or never ran. It posts a notification when the daemon stops being up.

Every heartbeat rewrites `~/.hands/status.json`, every effect and failure is a line
in `~/.hands/audit.jsonl`, and the daemon's output goes to `~/.hands/daemon.log`. Under launchd there is no terminal, so there is no key edge
yet: sessions are registered and spoken about but not answered by voice.

## Prior art

Happy (`~/code/happy`) does hands-free Claude Code over ElevenLabs ConvAI. Its core
workflow works and is why this approach is worth building; its trouble was transport
reliability, not content. Its two ideas worth keeping are the silent-context versus
speaks-now channel split and the read watermark. The full reading is in
[docs/happy-voice-reference.md](docs/happy-voice-reference.md).
