# Claude Code as the intermediary's brain

> Superseded by [control-point.md](control-point.md), which the `hands-wire-6ic` tickets build from.
> Where the two disagree (the proxy as a tap, `--bare`, the timing experiment as a first step, the ~300 ms
> requests, which are `count_tokens` calls and not model calls), it wins.

Written 2026-09-28 from a conversation between Brandon and Claude. The one-page version
is [overview.md](overview.md). This document holds every fact the conversation rested
on, where each came from, and what was decided against. Anything not verified is marked
as such.

## Summary

Hands' intermediary is the agent between the microphone and the working Claude Code
sessions. Today it is one stage of the Pipecat pipeline: an Anthropic or OpenAI LLM
service given the instruction in `src/hands/voice/intermediary_instruction.py` and the
twelve tools in `src/hands/voice/tools.py`. It cannot read a file, run a shell command,
or use a skill, and adding each of those as a Python function rebuilds a coding harness
one tool at a time.

The proposal replaces that one stage with a separate, slim Claude Code process that is
the intermediary: its own config directory, the same instruction as its system prompt,
read and bash, hands' twelve tools served over MCP, and skills written as SKILL.md. It
runs on the Claude subscription. A local reverse proxy between it and the API, grown
from cc-dump's proxy pipeline, gives hands the model's streamed text the moment it
leaves the API, so hands never reads Claude Code's rendered output. Everything else in
hands stays as it is.

## The picture

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [intermediary: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |   text deltas via the tap    v
                          +------------------------- [proxy] ---> api.anthropic.com
```

There are two Claude Code boxes and they stay two boxes.

The intermediary is one long-lived Claude Code process owned by the hands daemon. It
runs in the daemon's world, not in a project directory. It never has Edit, Write, or
Agent. It reaches the working sessions only through hands' tools: stage a draft, amend
it, send it, answer a permission or a question or a plan, run a slash command,
interrupt, list sessions, read what a session did.

The working sessions are Brandon's ordinary sessions in his ordinary setup, started any
way, in any terminal. Hands drives them as it does now: fritter types into them, hooks
and transcripts report back. Nothing about them changes.

What Brandon ruled out, in his words: "not user -> mic -> [single claude code session
that is both hands AND the instance of claude code that does the work]". And: "If I
wanted my own claude code to be driven by voice I can already do that. That isn't
cc-hands. That's just an MCP that reads claude's responses and speech to text."

## Why Claude Code feels slow, from the recordings

The evidence is in cc-dump's recordings at `~/.local/share/cc-dump/recordings/` (271
HAR files, 5.6 GB) and its `sessions.db`. Brandon's reading of them, which the numbers
support: Claude Code's perceived slowness is not the request-response cycle, it is that
Claude Code runs several cycles per user turn and curates the results.

From `sessions.db`, a session of 2026-02-13: one user turn at 00:34:19–00:35:04 produced
20 API requests. Most were a Haiku subagent carrying a 50 KB tool list, interleaved with
~1 KB Haiku utility calls. Session start alone was 10 Haiku calls in 10 seconds. The main
model's requests were 60–70 KB with 72 tools.

From the newest HAR, `ccdump-20260422-233238Z-2be307e3.har` (89 requests):

- Every main request is Opus with a 27.5 KB system prompt (the third system block alone
  is 27,574 characters), 67 tools (31 built-in plus 36 MCP, 30 of them one debugging
  server), and 60–390 messages. Cache reads run 100–200K tokens per call.
- A short-output round trip is 2.3–3.5 s wall-clock (entries with 91–181 output
  tokens). Long outputs scale with generation: 7,854 output tokens took 127 s.
- A stripped request (no system prompt, no tools, one message) returns in about 300 ms.
- Every request carries `metadata.user_id` of the form
  `user_<hash>_account_<uuid>_session_<uuid>`; the last part is the Claude Code session
  id, the same id hands' registry keys on.

So a single request with a cached prompt is fast even at Claude Code's full weight, and
a slim one is faster. The fan-out and the prompt weight are what a slim brain removes.

What the recordings cannot show: time to first delta. The HAR stores one `wait` time per
request and a reconstructed response body, with no per-event timestamps. For a voice
loop the first delta is the number that matters, because TTS starts on it. This is the
gap the first experiment closes.

## The brain: a slim Claude Code process

All flags below were read from `claude --help` on 2026-09-28.

**Isolation.** `CLAUDE_CONFIG_DIR` points at a hands-owned directory (the variable is
used throughout the Claude Code source; 24 references in the research fork). That
directory holds the brain's settings, skills, and login. None of Brandon's `~/.claude`
setup is loaded: `--setting-sources` narrows what is read, `--bare` skips hooks from
settings and installed plugins, LSP, and plugin sync. Because hands' own plugin hooks
would otherwise fire on the brain and hands would narrate itself, the brain must not
load that plugin; the separate config directory is what guarantees it.

**Prompt.** `--system-prompt` replaces Claude Code's prompt with the intermediary
instruction. `--exclude-dynamic-system-prompt-sections` keeps the rest out. The
instruction file stays the one source of truth: hands passes its contents at launch.

**Tools.** `--tools` names the built-in set: `Read`, `Bash`, `Skill`, and whatever else
is decided, with `""` disabling all. `--strict-mcp-config` with `--mcp-config` loads
only hands' MCP server, which serves the twelve session tools. Hands' tool bodies keep
their audit wrapper and their readbacks, which `src/hands/voice/readback.py` builds from
outcomes, never from the model repeating itself.

**Skills.** SKILL.md files in the brain's config directory. "File a lit ticket" and
"groom the backlog" become a skill plus shell, not Python.

**Transport.** `-p --input-format stream-json --output-format stream-json
--include-partial-messages`. The process stays up across turns; hands writes each
transcript as a user message on stdin. Claude Code's stdout stream is the authoritative
turn end and carries tool results. The proxy is the fast path for text.

**Auth.** The brain is Claude Code making Claude Code's own requests, so it runs on the
subscription. The proxy forwards those requests unchanged; it never makes requests of
its own.

**Where the patched build comes in.** For his own use, Brandon intends to run a
modified, deminified Claude Code. Flags cover the prompt, tools, hooks, and settings. Whatever fan-out the
flags do not reach (utility calls before the main request, if any survive) is what the
patched build cuts. The first experiment measures whether anything is left.

## The tap: a proxy grown from cc-dump

cc-dump (`~/code/cc-dump`, Python, ~27K lines) is a reverse proxy: set
`ANTHROPIC_BASE_URL` to it and Claude Code's traffic flows through, using Claude Code's
normal auth. Its `src/cc_dump/pipeline/` package is the reusable part; the `tui/`
package, most of the line count, is not.

What the tap keeps from cc-dump:

- `pipeline/response_assembler.py`: `ResponseAssembler` and
  `reconstruct_message_from_events`, which rebuild a message from SSE events.
- `pipeline/event_types.py`: the typed event model, with `parse_sse_event` as the one SSE
  validation boundary.
- `tui/stream_registry.py`: `_extract_session_id`, which reads the session id out of
  `metadata.user_id`, and the request-to-stream correlation around it.
- `pipeline/sentinel.py` as precedent: it answers a `$$`-prefixed user message with a
  synthetic SSE response without going upstream, so the proxy has already been an active
  participant once.

What the tap rewrites: the transport. cc-dump's server is threaded `http.server`
(`pipeline/proxy.py`, `ProxyHandler`, `_fan_out_sse` with `ClientSink` and
`EventQueueSink`). Hands is asyncio under Pipecat, so the tap is an asyncio proxy that
streams the upstream response to Claude Code byte for byte while feeding the same bytes
to the assembler.

What the tap does with what it sees:

- `text_delta` events from the brain's stream go to TTS as they arrive, before Claude
  Code parses them.
- A `tool_use` block starting in the stream is the earliest possible moment to say what
  the brain is doing ("filing the ticket").
- Every request is attributed to a session by the id in `metadata.user_id`.
- Utility calls are recognised by shape (tiny system prompt, no tools, Haiku) and not
  spoken.

Whether the tap lives in hands or becomes a shared package with cc-dump is Brandon's
call; the conversation leaned toward hands, because the tap is coupled to the registry
and to Pipecat.

## What changes in hands, precisely

Only the LLM stage of the Pipecat pipeline changes. Pipecat stays: mic, gate, VAD,
Whisper, the user and assistant turn aggregators, TTS, speakers, barge-in. Brandon's
words: "reaching for the stars here and not giving up all the magic that Pipecat
provides."

- `src/hands/voice/pipeline.py`: `build_llm` today constructs an `AnthropicLLMService`
  or `OpenAILLMService`. It becomes a processor that forwards the aggregated transcript
  to the brain process and emits the brain's streamed text as the same frames TTS
  consumes now. Barge-in, which today cancels the LLM service's generation, forwards an
  interrupt to the brain.
- `src/hands/voice/tools.py`: the twelve tool bodies stop being Pipecat
  `DirectFunction`s and become plain callables with schemas. Two thin adapters wrap
  them: the Pipecat shell that exists today, kept for the Python-core track, and an MCP
  server for the brain. The `audited` wrapper stays on the bodies.
- `src/hands/voice/intermediary_instruction.py`: unchanged as a file; it is passed to
  the brain as `--system-prompt`.
- `src/hands/daemon/run.py`: `HANDS_LLM` gains a variant that launches the brain
  process and the tap.
- A new tap module, per the section above.

What does not change: the `core` and `sessions` packages, fritter, the hook shims, the
audit log, readbacks, the narration tree, and everything Pipecat does on either side of
the LLM stage.

Two semantics the current stage has and the new one must keep:

- `stay_silent` returns with `run_llm=False`, so the choice not to answer produces no
  speech. With the brain, hands suppresses TTS for a turn in which that tool was called.
- The draft tools are marked `cancel_on_interruption=False` so a barge-in cannot change
  or send a draft without its readback being heard. Since the tool bodies run in hands'
  process, hands completes the effect regardless of the brain's interrupt and speaks the
  readback itself from the outcome.

## Optional: the working sessions through the same tap

Fritter's `claude` shim already sets environment variables on every session it wraps
(`FRITTER_SOCKET`, read at `src/hands/sessions/shim.py:90`). Setting
`ANTHROPIC_BASE_URL` there too routes every working session through the tap. Narration
would then have the model's live stream for each session, not only Stop hooks and
transcript tails.

This is observation only. It does not put the working sessions in the brain's process
or conversation. It is separable: if it is not wanted, nothing else in the proposal
changes.

## Alternatives considered, and why not

**The Claude Agent SDK.** Declined by Brandon "100% now and forever": it bills the API
rather than running on the subscription. Not to be proposed again, even as a comparison.

**Pi (`badlogic/pi-mono`).** Brandon raised it because it is far leaner than Claude
Code. Read from its docs on 2026-09-27: TypeScript; four built-in tools; skills in
Agent Skills format from `~/.agents/skills/`; an RPC mode over JSONL stdio with
`prompt`, `abort`, `steer`, `follow_up`, `get_state`, and events down to `text_delta`
and `tool_execution_start`; extensions that run in-process and are the only way to add
tools. Hands' twelve tools would have to be TypeScript extensions calling back into the
Python daemon over a socket, and Pi's OAuth support for Anthropic subscriptions is not
confirmed in its docs. The leanness is real; the language split and the tool boundary
are what ruled it out. What the conversation kept from Pi is its shape: tool loop,
read and bash, SKILL.md, compaction.

**A Python "Pi": a lean agent core.** This remains the public, long-term track for a
brain that calls the API with a key (pydantic-ai is the closest existing library; no
Python library discovered has a skills loader). It is a separate track from this
proposal, not a rejected one. Its loop is Pipecat's, which the pipeline already runs.

**Brandon's own Claude Code driven by voice.** Ruled out as not being cc-hands; see the
quote in "The picture".

**The proxy as the brain.** A proxy that answers the model's tool calls itself and makes
follow-up API requests with the OAuth token from Claude Code's headers. Ruled out for
two reasons: reusing that token from another client is what Anthropic was reported to
have blocked third-party harnesses for in 2025 (recollection, not re-checked), and it
is Brandon's account; and the proxy would then hold a
shadow message history spliced into every request, a second conversation state that
drifts from Claude Code's. The tap observes what one client sends and never makes
requests of its own.

## The problems this creates

1. **Two streams per turn.** An API 529 makes Claude Code retry, and the tap sees the
   response twice. Speaking from tap deltas means keying on request identity and
   dropping the aborted partial. cc-dump's `RequestRegistry` and `StreamRegistry` are
   the starting point.
2. **Utility calls share the session id.** Classification by shape is required, or hands
   narrates title generation.
3. **Barge-in must reach the brain.** Stopping TTS alone leaves the tool loop running
   silently. Claude Code's stream-json stdin has a control request for interrupt; not
   verified against the patched build.
4. **Cache stability.** The recordings show 30–200K cached tokens read per call. The
   brain's prompt must not vary between turns, or every turn pays the uncached price.
5. **Quota.** Every spoken question is a real Claude Code turn against the subscription
   window. Unquantified.
6. **Coupling.** The tap depends on Anthropic's SSE format, which is stable and
   documented, and on the `metadata.user_id` shape. The brain's request shape is pinned
   by the patched build.

## The first experiment

Before writing any hands code, about an hour:

1. Add a timestamp on the first SSE event to cc-dump's `_fan_out_sse`, so
   time-to-first-delta is recorded.
2. Run the patched Claude Code with `--system-prompt`, `--tools`, `--bare`,
   `--strict-mcp-config`, and `ANTHROPIC_BASE_URL` at cc-dump.
3. Ask three spoken-style questions.
4. Read two numbers from cc-dump: requests per turn, which should be 1, and
   milliseconds to first text delta.

One request per turn and about a second to first text proves the brain can run at
voice speed. Extra requests before the main one are exactly what the patched build
still has to cut, and cc-dump shows which. The expectation, extrapolated from the
recordings and not measured: a brain with a ~5 KB prompt, a dozen tools, and a short
conversation reaches first text well under the 2.3 s seen for full-weight Opus
requests, since a spoken reply is 30–60 output tokens.

## Not yet verified

- The stdin interrupt control request in Claude Code's stream-json mode, and how a
  half-finished turn lands in the session afterwards.
- Time to first delta for a slim brain; the HAR recordings cannot supply it.
- Whether `--tools` also governs MCP tools or only the built-in set; the help text says
  "from the built-in set".
- Pi's OAuth support for Anthropic subscriptions, should Pi be revisited.
