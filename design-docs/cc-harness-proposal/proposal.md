# Claude Code as the intermediary's brain

> This document is superseded by [control-point.md](control-point.md). The `hands-wire-6ic` tickets are based on
> control-point.md. Where the two documents disagree (the proxy as a tap, `--bare`, the timing experiment as a first
> step, the ~300 ms requests, which are `count_tokens` calls and not model calls), control-point.md takes precedence.

Written 2026-09-28 from a conversation between Brandon and Claude. A one-page version is
available at [overview.md](overview.md). This document contains every fact the
conversation relied on, the source of each fact, and the options that were rejected.
Unverified information is marked as unverified.

## Summary

The intermediary in hands is the agent between the microphone and the working Claude
Code sessions. Currently it is one stage of the Pipecat pipeline: an Anthropic or OpenAI
LLM service that uses the instruction in `src/hands/voice/intermediary_instruction.py`
and the twelve tools in `src/hands/voice/tools.py`. It cannot read a file, run a shell
command, or use a skill. Adding each of these capabilities as a Python function would
mean rebuilding a coding harness one tool at a time.

This proposal replaces that stage with a separate, slim Claude Code process that acts as
the intermediary. The process has its own config directory, uses the same instruction as
its system prompt, has read and bash tools, serves hands' twelve tools over MCP, and uses
skills written as SKILL.md. It runs on the Claude subscription. A local reverse proxy,
based on cc-dump's proxy pipeline, runs between the process and the API. The proxy
passes the model's streamed text to hands as soon as the API sends it, so hands never
reads Claude Code's rendered output. The rest of hands is unchanged.

## The picture

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [intermediary: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |   text deltas via the tap    v
                          +------------------------- [proxy] ---> api.anthropic.com
```

The diagram shows two Claude Code processes, and they remain separate.

The intermediary is a single long-running Claude Code process managed by the hands
daemon. It runs in the daemon's environment, not in a project directory. It never has
access to the Edit, Write, or Agent tools. It accesses the working sessions only through
hands' tools: stage a draft, amend it, send it, answer a permission or a question or a
plan, run a slash command, interrupt, list sessions, read what a session did.

The working sessions are Brandon's normal sessions in his normal setup. They can be
started in any way, in any terminal. Hands controls them in the same way it does now:
fritter types input into them, and hooks and transcripts report their activity back to
hands. The working sessions do not change.

Brandon ruled out the following design, in his words: "not user -> mic -> [single claude
code session that is both hands AND the instance of claude code that does the work]".
And: "If I wanted my own claude code to be driven by voice I can already do that. That
isn't cc-hands. That's just an MCP that reads claude's responses and speech to text."

## Why Claude Code feels slow, from the recordings

The evidence comes from cc-dump's recordings at `~/.local/share/cc-dump/recordings/`
(271 HAR files, 5.6 GB) and its `sessions.db` database. Brandon's interpretation, which
the numbers support, is that Claude Code's perceived slowness is not caused by the
request-response cycle itself. It is caused by Claude Code running several cycles per
user turn and then curating the results.

In `sessions.db`, a session from 2026-02-13 shows one user turn at 00:34:19–00:35:04 that
produced 20 API requests. Most were requests from a Haiku subagent with a 50 KB tool
list, interleaved with ~1 KB Haiku utility calls. Session startup alone made 10 Haiku
calls in 10 seconds. The main model's requests were 60–70 KB and included 72 tools.

From the newest HAR, `ccdump-20260422-233238Z-2be307e3.har` (89 requests):

- Every main request is Opus with a 27.5 KB system prompt (the third system block alone
  is 27,574 characters), 67 tools (31 built-in plus 36 MCP, 30 of them from one debugging
  server), and 60–390 messages. Each call reads 100–200K tokens from the cache.
- A short-output round trip takes 2.3–3.5 s wall-clock time (entries with 91–181 output
  tokens). Long outputs take longer in proportion to generation: 7,854 output tokens took
  127 s.
- A stripped request (no system prompt, no tools, one message) returns in about 300 ms.
- Every request includes a `metadata.user_id` of the form
  `user_<hash>_account_<uuid>_session_<uuid>`. The last part is the Claude Code session
  ID, which is the same ID that hands' registry uses as its key.

This shows that a single request with a cached prompt is fast even with Claude Code's
full prompt and tool set, and that a slim request is faster. A slim brain removes the
extra requests and the large prompt.

The recordings do not show the time to the first delta. The HAR stores one `wait` time
per request and a reconstructed response body, with no per-event timestamps. For a voice
loop, the time to the first delta is the most important number, because TTS starts when
the first delta arrives. The first experiment measures it.

## The brain: a slim Claude Code process

All flags below were read from `claude --help` on 2026-09-28.

**Isolation.** `CLAUDE_CONFIG_DIR` is set to a directory owned by hands (the variable is
used throughout the Claude Code source, with 24 references in the research fork). That
directory contains the brain's settings, skills, and login. None of Brandon's `~/.claude`
configuration is loaded: `--setting-sources` limits which settings are read, and `--bare`
skips hooks from settings and installed plugins, LSP, and plugin sync. The brain must not
load hands' own plugin. Otherwise, the plugin's hooks would run on the brain and hands
would narrate its own activity. The separate config directory ensures that the plugin is
not loaded.

**Prompt.** `--system-prompt` replaces Claude Code's prompt with the intermediary
instruction. `--exclude-dynamic-system-prompt-sections` excludes the remaining sections.
The instruction file remains the single source of truth: hands passes its contents to
the process at launch.

**Tools.** `--tools` specifies the built-in tools: `Read`, `Bash`, `Skill`, and any
others that are chosen later. `""` disables all built-in tools. `--strict-mcp-config`
with `--mcp-config` loads only hands' MCP server, which serves the twelve session tools.
Hands' tool bodies keep their audit wrapper and their readbacks.
`src/hands/voice/readback.py` builds the readbacks from outcomes, never from the
model repeating itself.

**Skills.** Skills are SKILL.md files in the brain's config directory. "File a lit
ticket" and "groom the backlog" are implemented as a skill plus shell commands, not as
Python code.

**Transport.** `-p --input-format stream-json --output-format stream-json
--include-partial-messages`. The process keeps running across turns. Hands writes each
transcript to stdin as a user message. Claude Code's stdout stream is the authoritative
signal for the end of a turn, and it includes tool results. The proxy provides a faster
path for text.

**Auth.** The brain is Claude Code sending its own normal requests, so it runs on the
subscription. The proxy forwards those requests unchanged and never sends requests of
its own.

**The patched build.** Brandon plans to run a modified, deminified build of Claude Code
for his own use. Flags control the prompt, tools, hooks, and settings. The patched build
removes any extra requests that the flags cannot disable (utility calls before the main
request, if any remain). The first experiment measures whether any remain.

## The tap: a proxy based on cc-dump

cc-dump (`~/code/cc-dump`, Python, ~27K lines) is a reverse proxy. When
`ANTHROPIC_BASE_URL` is set to it, Claude Code's traffic passes through the proxy, using
Claude Code's normal authentication. Its `src/cc_dump/pipeline/` package can be reused.
The `tui/` package, which makes up most of the line count, cannot.

The tap reuses the following parts of cc-dump:

- `pipeline/response_assembler.py`: `ResponseAssembler` and
  `reconstruct_message_from_events`, which rebuild a message from SSE events.
- `pipeline/event_types.py`: the typed event model, with `parse_sse_event` as the single
  place where SSE events are validated.
- `tui/stream_registry.py`: `_extract_session_id`, which reads the session ID from
  `metadata.user_id`, and the surrounding code that matches requests to streams.
- `pipeline/sentinel.py`, as a precedent: it responds to a user message that starts with
  `$$` with a synthetic SSE response, without forwarding the request upstream. This shows
  that the proxy has already acted as an active participant in one case.

The tap replaces the transport. cc-dump's server uses the threaded `http.server` module
(`pipeline/proxy.py`, `ProxyHandler`, `_fan_out_sse` with `ClientSink` and
`EventQueueSink`). Hands uses asyncio under Pipecat, so the tap is an asyncio proxy. It
streams the upstream response to Claude Code byte for byte and passes the same bytes to
the assembler.

The tap processes the traffic as follows:

- `text_delta` events from the brain's stream are sent to TTS as they arrive, before
  Claude Code parses them.
- The start of a `tool_use` block in the stream is the earliest point at which hands can
  announce what the brain is doing (for example, "filing the ticket").
- Every request is assigned to a session by the ID in `metadata.user_id`.
- Utility calls are identified by their shape (tiny system prompt, no tools, Haiku) and
  are not spoken.

Brandon will decide whether the tap is part of hands or a package shared with cc-dump.
The conversation favored placing it in hands, because the tap depends on the registry
and on Pipecat.

## What changes in hands, precisely

Only the LLM stage of the Pipecat pipeline changes. The rest of Pipecat remains: mic,
gate, VAD, Whisper, the user and assistant turn aggregators, TTS, speakers, barge-in. In
Brandon's words: "reaching for the stars here and not giving up all the magic that
Pipecat provides."

- `src/hands/voice/pipeline.py`: `build_llm` currently constructs an
  `AnthropicLLMService` or `OpenAILLMService`. It is replaced by a processor that
  forwards the aggregated transcript to the brain process and emits the brain's streamed
  text as the same frames that TTS uses now. Barge-in currently cancels the LLM
  service's generation; instead, it will send an interrupt to the brain.
- `src/hands/voice/tools.py`: the twelve tool bodies are no longer Pipecat
  `DirectFunction`s; they become plain callables with schemas. Two thin adapters wrap
  them: the existing Pipecat adapter, kept for the Python-core track, and an MCP server
  for the brain. The tool bodies keep the `audited` wrapper.
- `src/hands/voice/intermediary_instruction.py`: unchanged as a file; it is passed to
  the brain as `--system-prompt`.
- `src/hands/daemon/run.py`: `HANDS_LLM` gets a new option that launches the brain
  process and the tap.
- A new tap module, as described in the previous section.

What does not change: the `core` and `sessions` packages, fritter, the hook shims, the
audit log, readbacks, the narration tree, and everything Pipecat does on either side of
the LLM stage.

The new stage must keep two behaviors of the current stage:

- `stay_silent` returns with `run_llm=False`, so when the intermediary chooses not to
  answer, no speech is produced. With the brain, hands suppresses TTS for any turn in
  which that tool was called.
- The draft tools are marked `cancel_on_interruption=False`, so a barge-in cannot change
  or send a draft without the user hearing its readback. Because the tool bodies run in
  hands' process, hands completes the tool's action even if the brain is interrupted, and
  speaks the readback itself based on the result.

## Optional: the working sessions through the same tap

Fritter's `claude` shim already sets environment variables on every session it wraps
(`FRITTER_SOCKET`, read at `src/hands/sessions/shim.py:90`). Also setting
`ANTHROPIC_BASE_URL` in the shim would route every working session through the tap.
Narration would then have the model's live stream for each session, in addition to Stop
hooks and the ends of transcripts.

This option is for observation only. It does not move the working sessions into the
brain's process or conversation. It is independent of the rest of the proposal: if it is
not adopted, nothing else in the proposal changes.

## Alternatives considered, and why not

**The Claude Agent SDK.** Brandon rejected it "100% now and forever" because it bills
API usage instead of running on the subscription. Do not propose it again, even as a
comparison.

**Pi (`badlogic/pi-mono`).** Brandon suggested it because it is much smaller than Claude
Code. From its docs, read on 2026-09-27: TypeScript; four built-in tools; skills in
Agent Skills format from `~/.agents/skills/`; an RPC mode over JSONL stdio with
`prompt`, `abort`, `steer`, `follow_up`, `get_state`, and events down to `text_delta`
and `tool_execution_start`; extensions that run in-process and are the only way to add
tools. Hands' twelve tools would have to be TypeScript extensions that call back into the
Python daemon over a socket, and Pi's docs do not confirm OAuth support for Anthropic
subscriptions. Pi is genuinely smaller, but it was ruled out because of the split between
two languages and because tools can be added only as in-process extensions. The
conversation kept Pi's overall design: a tool loop, read and bash, SKILL.md, and
compaction.

**A Python "Pi": a lean agent core.** This remains the public, long-term track for a
brain that calls the API with an API key (pydantic-ai is the closest existing library;
none of the Python libraries found has a skills loader). It is a separate track from
this proposal, not a rejected alternative. Its agent loop is Pipecat's loop, which the
pipeline already runs.

**Brandon's own Claude Code driven by voice.** Ruled out because it is not cc-hands; see
the quote in "The picture".

**The proxy as the brain.** In this design, a proxy handles the model's tool calls itself
and makes follow-up API requests with the OAuth token from Claude Code's headers. It was
ruled out for two reasons. First, Anthropic was reported to have blocked third-party
harnesses in 2025 for reusing that token from another client (from memory, not
re-checked), and the account at risk is Brandon's. Second, the proxy would have to keep a
separate message history and insert it into every request, which creates a second
conversation state that diverges from Claude Code's. The tap only observes what one
client sends and never makes requests of its own.

## The problems this creates

1. **Two streams per turn.** When the API returns a 529, Claude Code retries, and the tap
   sees the response twice. To speak from tap deltas, hands must track each request's
   identity and discard the partial response from the aborted request. cc-dump's
   `RequestRegistry` and `StreamRegistry` are the starting point.
2. **Utility calls share the session ID.** Hands must classify requests by shape;
   otherwise, it narrates title generation.
3. **Barge-in must reach the brain.** Stopping TTS alone leaves the tool loop running
   with no audio output. Claude Code's stream-json stdin supports a control request for
   interrupts; this has not been verified with the patched build.
4. **Cache stability.** The recordings show 30–200K cached tokens read per call. The
   brain's prompt must stay the same across turns; otherwise, every turn pays the price
   of an uncached prompt.
5. **Quota.** Every spoken question is a full Claude Code turn that counts against the
   subscription usage window. The usage has not been quantified.
6. **Coupling.** The tap depends on Anthropic's SSE format, which is stable and
   documented, and on the format of `metadata.user_id`. The patched build keeps the
   format of the brain's requests fixed.

## The first experiment

Run this experiment before writing any hands code. It takes about an hour:

1. Add a timestamp on the first SSE event to cc-dump's `_fan_out_sse`, so
   time-to-first-delta is recorded.
2. Run the patched Claude Code with `--system-prompt`, `--tools`, `--bare`,
   `--strict-mcp-config`, and `ANTHROPIC_BASE_URL` set to cc-dump.
3. Ask three questions phrased as they would be spoken.
4. Read two numbers from cc-dump: requests per turn, which should be 1, and
   milliseconds to first text delta.

If each turn makes one request and the first text arrives in about one second, the brain
is fast enough for voice. Any extra requests before the main request are the ones the
patched build still has to remove, and cc-dump shows which requests they are. The
expected result, extrapolated from the recordings and not measured: a brain with a ~5 KB
prompt, a dozen tools, and a short conversation returns its first text well under the
2.3 s observed for full-weight Opus requests, because a spoken reply is 30–60 output
tokens.

## Not yet verified

- The stdin interrupt control request in Claude Code's stream-json mode, and how a
  partially completed turn is recorded in the session afterward.
- Time to first delta for a slim brain; the HAR recordings cannot provide it.
- Whether `--tools` also controls MCP tools or only the built-in set; the help text says
  "from the built-in set".
- Pi's OAuth support for Anthropic subscriptions, if Pi is reconsidered.
