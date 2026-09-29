# Claude Code as the intermediary's brain: overview

Written 2026-09-28 from a conversation between Brandon and Claude. The full version,
with every number and file path, is [proposal.md](proposal.md).

## The idea in one paragraph

The intermediary that sits between the microphone and the working Claude Code sessions
is today a Pipecat LLM stage with twelve hand-written tools. Replace that one stage with
a slim, separate Claude Code process: its own config directory, the intermediary's
system prompt, a few built-in tools, hands' session tools served over MCP, and skills
written as SKILL.md. It runs on the Claude subscription, it can read a file, run `lit`, and load
a skill, and nothing else in hands changes. A local proxy between that process and the
API, grown from cc-dump, hands the model's streamed text straight to TTS, so hands never
reads Claude Code's output.

## The picture

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [intermediary: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |   text deltas via the tap    v
                          +------------------------- [proxy] ---> api.anthropic.com
```

Two Claude Code boxes, kept apart on purpose. The intermediary is one long-lived process
that never runs in a project directory, never edits code, and reaches the working
sessions only through hands' tools (stage a draft, send it, answer a permission, read
what a session did). The working sessions are Brandon's normal sessions, driven and
observed exactly as they are now.

## Why this and not the alternatives

Claude Code feels slow because one user turn fans out into many API requests: a Feb 13
recording shows one turn becoming 20 requests, most of them a Haiku subagent and small
utility calls. A single request with a cached prompt returns in about 2.5 seconds even
carrying a 27 KB system prompt and 67 tools. Strip the fan-out and the prompt and the
harness is fast; every knob needed is a `claude` flag.

The Claude Agent SDK was declined permanently because it bills the API instead of using
the subscription. Pi is TypeScript, and its RPC mode cannot register tools from outside.
Driving Brandon's own Claude Code by voice is not cc-hands. A proxy that makes its own
API calls with Claude Code's OAuth token risks the account and keeps a second
conversation history.

## What changes in hands, and what does not

Changes: the LLM stage in `src/hands/voice/pipeline.py` becomes a processor that
forwards the transcript to the brain and emits its streamed text as frames; the tool
bodies in `src/hands/voice/tools.py` get an MCP adapter beside the Pipecat one; a tap
module borrows cc-dump's SSE assembler and event types.

Does not change: everything else in Pipecat (mic, VAD, Whisper, turn aggregation, TTS,
barge-in), the `core` and `sessions` packages, fritter, hooks, the audit log, and the
readbacks, which are built from outcomes rather than model words.

## The first step

Before writing any hands code: run a slim Claude Code (`--system-prompt`, `--tools`,
`--bare`, `--strict-mcp-config`) through cc-dump with a first-SSE-event timestamp
added, ask three spoken-style questions, and read two numbers: requests per turn
(should be 1) and milliseconds to first text delta. Those decide whether Claude Code can
be the brain at voice speed.
