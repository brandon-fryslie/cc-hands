# Claude Code as the intermediary's brain: overview

> This document is superseded by [control-point.md](control-point.md). The `hands-wire-6ic` tickets are based on
> control-point.md. Where the two documents disagree (the proxy as a tap, `--bare`, the timing experiment as a first
> step, the ~300 ms requests, which are `count_tokens` calls and not model calls), control-point.md takes precedence.

Written 2026-09-28 from a conversation between Brandon and Claude. The full version,
with every number and file path, is [proposal.md](proposal.md).

## The idea in one paragraph

The intermediary sits between the microphone and the working Claude Code sessions. It
is currently a Pipecat LLM stage with twelve hand-written tools. This proposal replaces
that stage with a separate, slim Claude Code process that has its own config directory,
the intermediary's system prompt, a few built-in tools, hands' session tools served over
MCP, and skills written as SKILL.md. The process runs on the Claude subscription and can
read a file, run `lit`, and load a skill. No other part of hands changes. A local proxy
based on cc-dump runs between that process and the API and sends the model's streamed
text directly to TTS. As a result, hands never reads Claude Code's output.

## The picture

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [intermediary: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |   text deltas via the tap    v
                          +------------------------- [proxy] ---> api.anthropic.com
```

The diagram shows two Claude Code processes, and they are kept separate on purpose. The
intermediary is a single long-running process. It never runs in a project directory,
never edits code, and accesses the working sessions only through hands' tools (stage a
draft, send it, answer a permission, read what a session did). The working sessions are
Brandon's normal sessions. Hands controls and monitors them in the same way it does now.

## Why this and not the alternatives

Claude Code feels slow because each user turn results in many API requests. In a
recording from Feb 13, one turn produced 20 requests, most of them from a Haiku subagent
and small utility calls. A single request with a cached prompt returns in about 2.5
seconds, even with a 27 KB system prompt and 67 tools. If the extra requests and the
large prompt are removed, the harness is fast. Every setting this requires is available
as a `claude` flag.

The Claude Agent SDK was permanently rejected because it bills API usage instead of
using the subscription. Pi is written in TypeScript, and its RPC mode cannot register
external tools. Controlling Brandon's own Claude Code by voice is outside the scope of
cc-hands. A proxy that makes its own API calls with Claude Code's OAuth token puts the
account at risk and maintains a second conversation history.

## What changes in hands, and what does not

Changes: the LLM stage in `src/hands/voice/pipeline.py` is replaced by a processor that
forwards the transcript to the brain and emits the brain's streamed text as frames; the
tool bodies in `src/hands/voice/tools.py` get an MCP adapter in addition to the existing
Pipecat adapter; a new tap module reuses cc-dump's SSE assembler and event types.

Does not change: everything else in Pipecat (mic, VAD, Whisper, turn aggregation, TTS,
barge-in), the `core` and `sessions` packages, fritter, hooks, the audit log, and the
readbacks, which are generated from outcomes rather than from model output.

## The first step

Before writing any hands code, add a timestamp for the first SSE event to cc-dump, run a
slim Claude Code (`--system-prompt`, `--tools`, `--bare`, `--strict-mcp-config`) through
cc-dump, and ask three questions phrased as they would be spoken. Then record two
numbers: requests per turn (expected: 1) and milliseconds to the first text delta. These
numbers determine whether Claude Code is fast enough to serve as the brain for voice
interaction.
