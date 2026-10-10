# The proxy as the point of control

Written 2026-09-29 from a conversation between Brandon and Claude. This document
supersedes the earlier documents in this directory, which describe the proxy as a
passive tap that only observes traffic. The proxy is not a tap. It is the point of
control on the connection between the intermediary's brain and the API, in both
directions, and this changes what hands can do.

## Architecture overview

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [brain: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |    A: every model input       v
                          |       and output         [proxy: hands]  <->  api.anthropic.com
                          +--------------------------------+
```

There are two separate kinds of Claude Code process. The brain is a single long-lived,
slim Claude Code process that the hands daemon owns. It consists of the intermediary's
system prompt, a few built-in tools, hands' session tools over MCP, and skills as
SKILL.md files. It runs on the Claude subscription because the requests are Claude
Code's own requests. It never runs in a project directory and never edits code. The
working sessions are Brandon's ordinary sessions, which hands drives and observes in the
same way as today. In Brandon's words: "not user -> mic -> [single claude code session
that is both hands AND the instance of claude code that does the work]".

The Pipecat pipeline stays intact. The mic, gate, VAD, Whisper, turn aggregation, TTS,
speakers, and barge-in do not change. Only the LLM stage of the pipeline changes:
`build_llm` in `src/hands/voice/pipeline.py` no longer constructs an Anthropic or OpenAI
service. Instead, it becomes a processor that sends the transcript to the brain and
emits the response from the wire as the frames that TTS already consumes. Brandon:
"reaching for the stars here and not giving up all the magic that Pipecat provides."

## Two sources, one rule

Hands can get information about the brain from two sources:

- **A, the wire.** Every request the brain makes and every byte the API returns pass
  through the proxy, with the request before the response. A is complete: all model
  input and output passes through it.
- **B, the harness.** Claude Code's stdout stream, its stderr, its exit code, and the
  hooks it fires. B reports what the harness did around the model: when a tool ran,
  the result of a permission decision, and whether the process exited.

The rule: primary facts come from A, derived facts come from B, and no fact comes from
both. Hands generates speech from A. B is used only for information that only the
harness has. Reading the same fact from both A and B does not give two independent
streams. It gives the same stream observed at two different times, and processing it
twice is a bug.

A is not only earlier than B. A is also complete, and B is not: B shows only what the
harness chooses to expose, while A shows all model input and all model output.

### Apparent problems from the tap's perspective

Each of these issues was a concern when the proxy was designed as a passive observer.
Now that the proxy is the point of control, each one is a policy decision.

**Retries.** The proxy sees a 529 error or a dropped upstream stream before Claude Code
does. It passes the error through and recognizes Claude Code's byte-identical retry
request, so hands can tell the attempts apart. If the proxy retried upstream itself, it
would be sending its own request with Claude Code's credentials, which it never does
(see below).

**Compaction and utility calls.** The proxy sees each request before any response bytes
exist. A utility call has no tools and uses a different model. A compaction is a fork
(see below), so it shares the brain's prefix in the same way as a main turn. How to
identify a compaction will be determined through use. Requests are classified on the
way out, so the response is already tagged when it arrives. The classification does not
need to be complete at the start. Brandon: "Simply using the application should reveal
this quickly with no effort." The one requirement from the first version is the default
behavior: a request with an unrecognized shape is never spoken, and it is logged at a
high severity.

**Refused or failed tools.** This information is also on A: the tool result, including
its error flag, is included in the messages of the next request. B adds the time the
tool ran, not whether it ran.

**Turn end.** The harness reports this with a simple signal: the stdout result event, a
hook, or the exit code. The harness cannot exit without a signal, because the exit
itself is the signal.

## Modifying the request

Because the proxy composes the input that the model receives, hands is no longer limited
to the content that Claude Code puts in the conversation. Prompt caching limits how far
this can go. (The caching behavior described below is documented API behavior. It was
not re-tested on the date this document was written.)

The cache matches an exact prefix up to a breakpoint. A change at byte N invalidates
everything after byte N. As a result:

- **The stable body is byte-identical from turn to turn.** This includes the
  instructions, the tool schemas, and the history. The proxy never rewrites the
  beginning or the middle of the request on a per-turn basis.
- **Per-turn content is added only at the end.** Session status and new notes from hands
  (the appended content) are appended to, or placed inside, the newest user message. On
  the next turn, Claude Code's history does not contain the content that the proxy
  added. If the content is appended after the block that holds Claude Code's breakpoint
  on that message, it is outside the cached prefix and the next request gets a full
  cache hit. If it is placed anywhere earlier, the rest of the message after it is
  uncached once, and during a tool loop that message is a tool result, which can be any size. This also means that
  outdated notes do not accumulate in the history the way `[hands]` messages do today.
  If zero cache misses are ever wanted, the proxy re-inserts its own past
  insertions, computed as a pure function of its own log.
- **History is edited only in batches.** Trimming an old tool result invalidates the
  prefix from that point, so trims are done every K turns, all at once, with one cache
  rebuild accepted. Claude Code continues to send the untrimmed history, so each trim is
  applied again to every later request, computed as a pure function of the proxy's own
  log. Compaction works the same way for the same reason.
- **Claude Code already uses the caching layout we need.** In the newest recording, the
  system prompt includes `cache_control: {type: ephemeral, ttl: 1h}` and a third
  breakpoint is on the latest user message. The proxy can move or add breakpoints if the
  appended content ever needs a breakpoint in a different position, but this is not
  required.

The appended content is the product surface. The content that hands writes there
each turn replaces today's start-up note and `[hands]` messages. It is generated from
the registry on every turn instead of being accumulated.

## Managing the brain's context

The brain cannot summarize its own context while it is generating a response, the proxy
does not make its own requests, and the rule against using Claude Code's OAuth token from
another client still applies. There are three approaches, listed in order of preference:

**Rule-based trimming, with no model.** The brain's context has a predictable structure:
short spoken turns, temporary notes in the appended content, and tool results, which make up most of
the volume. A `read_session` result from twenty minutes ago is replaced by a one-line
stub according to a rule, in the pure core, in batches every K turns.

**Forks, which Claude Code already supports.** `/btw` is implemented in
`src/utils/forkedAgent.ts` of the research fork as a forked agent. A forked agent is a
second query loop that reuses the main loop's exact system prompt, context, tools, and
messages (the "cache-safe params, must match parent for cache hits"), appends a side
question, and never writes its messages into the main conversation. The same mechanism
runs `postTurnSummary`, `promptSuggestion`, `session_memory`, and `compact`. This
provides the side request that we thought the brain did not have: it is permitted under
the subscription, it shares the brain's prompt cache so that it costs only the appended
content and the output, and it leaves nothing in the history. On the wire, a fork shares
the brain's prefix in the same way as a main turn. How to distinguish the two is not
verified. `/btw` itself is an interactive command, and whether a fork can be triggered
from stdin in `-p` mode is not verified. If it cannot, adding a way to trigger one is one
of the two tasks for the patched build (see "Required changes to Claude Code").

**Steering compaction.** Claude Code's own compaction is a fork, so its requests go
through the proxy. Instead of disabling compaction, the proxy recognizes it on the way
out and rewrites its summarization prompt to keep the information that a voice session
needs (session titles, the draft in progress, and decisions made). The response then
replaces Claude Code's history. This costs one cache rebuild, under our control.

The local model that hands already has is kept as a fallback for summarization outside
the subscription. The design does not depend on it.

## The summary store

Brandon asked whether a cache of summaries of individual pieces of content, indexed and
invalidated by hash, is worth building, or whether reads are cheap enough that the cache
is not worth the effort. Measurements of the links backlog answered the question. These
figures were collected without reading the content:

- 132 open tickets. `lit show` for all of them produces 7,149 lines, 579 KB, about 145K
  tokens.
- The raw descriptions alone are 184 KB, about 46K tokens, with a median of 1.3 KB per
  ticket.
- The whole tracker, including 634 closed tickets, is 1.4 MB, about 358K tokens.
- The listing alone is 11 KB, about 2.8K tokens.

To answer "What's in the backlog?" by reading the raw tickets, the brain would need more
context than it can hold. Even where it could hold it, each question would require tens
of seconds of uncached reading. The data is large, changes often, and is read
frequently. The store is therefore worth building, with this design:

- **Keyed by content hash** of the title and description, with the summarizer's version
  included in the key. The key does not use `updated_at`, because a rerank changes that
  field without changing the ticket's content and should not trigger a new summary.
- **One sentence per ticket.** 132 × ~25 tokens is about 3K tokens. The whole backlog
  fits in a context sized for spoken interaction, and the raw ticket is available with
  one call.
- **Layered.** A parent's summary is keyed by the hash of its own text and its children's
  summaries. This continues up to the backlog, whose children are its epics and the
  tickets that are not in an epic. When one ticket is edited, only that ticket and its
  ancestors are recomputed. A question such as "How's the observability epic going?"
  becomes a lookup.
- **Summarized outside the voice path.** Summaries are generated when content is first
  seen and when it changes, never while the user is waiting, by a fork of the brain or by
  the local model.
- **One table for all content-addressed data.** Files are content-addressed by nature. A
  finished turn of a work session is immutable and is keyed by its ID. The per-turn
  summaries that hands already produces for narration go in the same store, so
  `read_session` returns summaries first and raw content on request. This puts a few
  hundred tokens in the brain's context instead of thousands.

This store also provides the foundation that the endurance epic (`hands-memory-5qk`)
called for. Recall becomes a search over existing summaries, and the store is append-only
where the content is immutable.

The resulting change in hands: a `summaries` store, and `read_backlog`, `read_ticket`, and
`read_session` tools that send pointers and retrieve content on request, which is the
design that `docs/architecture.md` already describes.

On lit's `prompt` field, in Brandon's words: "an idea that essentially never quite became
necessary. The entire output of every command is a prompt, in essence."

## What Claude Code adds to each request

The following data is from the newest recording (`ccdump-20260422-233238Z-2be307e3.har`,
Claude Code 2.1.111). It lists what the slim brain removes:

- System prompt: three blocks of 81, 57, and 27,574 characters. The last two have 1h
  cache breakpoints. Additional body fields: `thinking: adaptive`,
  `context_management` (which clears old thinking), `output_config`, and
  `metadata.user_id`, which contains the device, account, and session IDs.
- First user message: 509 characters typed by the user and 47,889 characters injected as
  `<system-reminder>` blocks. The injected blocks are the skills list (15K), CLAUDE.md and
  context (30K), deferred tools (2K), and SessionStart hook output (1K).
- Later turns: 0–1.1K characters injected on a typical turn, and 45K on a turn that
  includes a Read result with plan-mode reminders. Fifty-seven tool results over the
  session totaled 171K characters.
- The ~300 ms requests match the format of `count_tokens` calls: they have no
  `max_tokens`, the only message is a file's contents, and the response is
  `{"input_tokens": N}`. Claude Code uses them to measure the size of a file before
  adding it to the context. They are not model calls, but they are round trips, and they
  occur in groups of four.

## Required changes to Claude Code

Fewer changes are needed than expected, but some are required. From `claude --help` on
2026-09-28: `--system-prompt` replaces the prompt; `--tools` specifies the built-in tool
set, and `""` disables it; `--strict-mcp-config` with `--mcp-config` loads only hands'
server; `--setting-sources` limits which settings are read; `-p --input-format
stream-json --output-format stream-json --include-partial-messages` is the transport;
`CLAUDE_CONFIG_DIR` isolates the brain's config, skills, and login; and
`ANTHROPIC_BASE_URL` routes the brain's traffic through the proxy, which cc-dump
demonstrates in daily use. An interrupt over stdin is a control request that the CLI's
stream-json protocol supports for its own SDK. Whether it works with a slim `-p` brain
is listed below as unverified.

Superseded 2026-09-30 (hands-wire-6ic.99l): the brain does not use `-p`. Brandon: no
`claude -p` anywhere, and no request is sent that Claude Code did not send itself. The
brain runs in interactive mode under hands' own fritter. A turn is typed into its input,
a fork is `/btw` typed the same way, an interrupt is the Escape key, and turn ends are
reported by its UserPromptSubmit, Stop, and StopFailure hooks. The brain uses
`--append-system-prompt` instead of `--system-prompt`, because the API checks that the
system prompt begins the same way as Claude Code's.

Superseded 2026-09-30 (hands-wire-6ic.gnk): only a turn and its stop keys are typed into
the brain. A side question is not a fork of the brain. Each side question is the opening
prompt (`claude "/btw ..."`) of a separate interactive Claude Code process with no
tools, started only for that question (`src/hands/brain/asides.py`).

`--bare` cannot be used. Brandon knew this from experience, and the help text and the
source confirm it: "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via
--settings (OAuth and keychain are never read)", and `auth.ts:101` in the research fork:
"--bare: API-key-only, never OAuth." The `CLAUDE_CODE_SIMPLE` variable enables the same
mode (`isBareMode()` in `envUtils.ts:60` checks the variable or the flag), so there is
no workaround. Simple mode would also disable more than needed: the tools are limited to
Bash, Read, and Edit, and the system prompt is reduced to three lines. The features that
`--bare` would have disabled therefore need to be handled separately: hooks and plugin
sync by using a config directory that has none; CLAUDE.md discovery by using a working
directory that has none; and auto-memory, LSP, and background prefetches by their own
settings, which have not yet been identified.

The patched build therefore has two specific tasks, each small: separate OAuth from bare
mode in `auth.ts`, about four lines, if the per-feature settings are not sufficient; and
make a fork available in headless mode, if one cannot be triggered from stdin. Everything
else is done with flags. For any utility request that remains, there is also an option
that needs no patch: the proxy responds to the request itself and does not forward it
upstream, which cc-dump's `sentinel.py` already does for `$$` messages. This rewrites a
response to the client, not a request to Anthropic.

## Timing is measured by the proxy, not by an experiment

An earlier draft scheduled a timing experiment as the first step. The experiment is not
a prerequisite, and it is not a ticket. Brandon's position on latency, which the
recordings support: the same bytes on the wire take the same time regardless of whether
Claude Code or another client sent them. A short-output Opus request with a 27 KB
prompt, 67 tools, and 100–200K cached tokens returns in 2.3–3.5 s. The recordings also
already show how many requests a turn makes: more than one, and the previous section
lists which ones. The recordings do not include timing outside the request, and no value
of that timing would change the design:

- stdin write → the request leaves Claude Code: if this is large, it is fixed by a
  setting or a patch in the brain's launch, not by a design change;
- request leaves → first SSE byte: this is the API's minimum latency, which is the same
  in any design;
- first SSE byte → the same text on Claude Code's stdout: this would matter only if it
  determined whether hands reads the wire or stdout, and that is determined by
  completeness;
- requests per turn: handled by the classification default and determined through use.

The proxy therefore records timestamps for every request on the wire from its first
version (request in, request out, first byte, last byte), and those measurements become
available the first time the brain answers a question. The stdin and stdout endpoints
are not on the wire; the brain's launcher observes them. The wire measurements are the
acceptance check for the brain launch, so the measurement necessarily happens last.

## Implementation plan

Filed in lit on 2026-09-29 as epic `hands-wire-6ic`, with the tickets below as its
children, in this order. Another session can recreate the backlog from this section
alone, and each ticket references this document by path. Question tickets are placed
before the work that depends on them, so that a developer does not need to stop and
investigate during implementation.

1. **The proxy.** An asyncio reverse proxy in hands that Claude Code connects to through
   `ANTHROPIC_BASE_URL`. It forwards each request upstream unchanged, streams the
   response back byte for byte, and sends the same bytes to an assembler ported from
   cc-dump's `response_assembler.py` and `event_types.py`. Every request is associated
   with a session by `metadata.user_id`, classified on the way out (main turn, fork,
   compaction, count_tokens, unknown), timestamped at request-in, request-out,
   first-byte, and last-byte, and written to the audit log. Requests with unknown shapes
   are never spoken and are logged at a high severity. The proxy never makes its own
   requests and never reuses the client's credentials. Purpose: to provide all traffic
   between one Claude Code process and the API, in both directions, as typed events.
   Second consumer: the working sessions (ticket 10).
2. **Question: which settings replace `--bare`.** `--bare` and `CLAUDE_CODE_SIMPLE`
   both disable OAuth. Find the settings or environment variables that disable
   auto-memory, LSP, background prefetches, and CLAUDE.md discovery for a `-p` process
   while OAuth stays enabled. If no setting exists for a feature, record which feature,
   so that the brain-launch ticket can determine whether the `auth.ts` patch is needed.
   The answer is recorded by updating ticket 4.
3. **Question: interrupt over stdin.** In `-p --input-format stream-json` mode, send the
   interrupt control request during a turn and record how Claude Code responds: whether
   generation stops, what is written to the session history, and what is printed to
   stdout. The answer is recorded by updating ticket 5.
4. **The brain launch.** A `HANDS_LLM` variant in `src/hands/daemon/run.py` that starts
   one long-lived slim Claude Code process: `CLAUDE_CONFIG_DIR` set to a hands-owned
   directory, `--system-prompt` from `src/hands/voice/intermediary_instruction.py`,
   `--tools` set to the built-in tool set defined there, `--strict-mcp-config` with
   hands' MCP server, `-p --input-format stream-json --output-format stream-json
   --include-partial-messages`, `ANTHROPIC_BASE_URL` set to the proxy, and the settings
   that ticket 2 identified. The MCP server is a second adapter over the tool
   implementations in `src/hands/voice/tools.py`. These change from Pipecat
   `DirectFunction`s to plain callables with schemas. The Pipecat wrapper remains as the
   first adapter, and the `audited` wrapper remains on the implementations. Done when
   the brain answers a question typed over stdin, its text is visible on the proxy, and
   the four timestamps for that turn are in the audit log.
5. **The LLM stage.** `build_llm` in `src/hands/voice/pipeline.py` gets a processor that
   sends the aggregated transcript to the brain's stdin and emits the text deltas from
   the wire as the frames that TTS consumes today. Barge-in forwards the interrupt (as
   determined by ticket 3). `stay_silent` keeps its current meaning: nothing is spoken
   after it (`run_llm=False` in Pipecat). Claude Code sends every tool result back for
   another response, so with the brain, the LLM stage is responsible for suppressing
   that response. The draft tools complete their action even if an interrupt occurs,
   and their readback is spoken based on the result. Done when the user asks a
   question by voice and hears the answer, and the audit log shows that the text came
   from the wire.
6. **The appended content.** The content that hands appends to the newest user message
   each turn, generated from the registry: session status, and the information that the
   start-up note and `[hands]` messages contain today. The stable body is not changed.
   Because the content is placed after the block that holds Claude Code's breakpoint, it
   does not affect the cache (see "Modifying the request").
7. **The summary store.** A store keyed by content hash plus summarizer version, with one
   sentence per item. It is layered so that a parent's key is the hash of its own text
   and its children's summaries. The `read_backlog`, `read_ticket`, and `read_session`
   tools return summaries first and raw content on request. Summarization runs outside
   the voice path. Purpose: to provide a one-sentence summary for any content-addressed
   item, kept until the item changes. Second consumer: narration's per-turn summaries,
   and lit itself if the summaries are written back. Sizing figures: 132 open tickets,
   145K tokens through `lit show`.
8. **Question: can a fork be triggered in headless mode.** `/btw` runs a forked agent
   (`src/utils/forkedAgent.ts` in the research fork) that shares the brain's prompt
   cache and leaves nothing in the history. Determine whether a `-p` process can be
   instructed to start one from stdin. The answer is recorded by updating ticket 9.
9. **Forks and compaction steering.** Long-term: the brain's context is managed by
   rule-based trimming in batches, forks for summaries, and recognition of Claude Code's
   compaction on the way out with its prompt rewritten. This ticket cannot be scoped
   until ticket 8 determines whether forks are available. If they are not, the patched
   build's second task is to make one available.
10. **Work sessions through the proxy.** Fritter's `claude` shim sets
    `ANTHROPIC_BASE_URL` (next to `FRITTER_SOCKET`, which is set in
    `fritter/main.go:70`) so that every wrapped session's traffic goes through the
    proxy, and narration receives the model's stream instead of only Stop hooks. This
    is observation only and can be done independently.

The epic's end-to-end acceptance test: the user asks by voice what a session is doing,
what is in the backlog, and to file a ticket, and hears each answer. The user
interrupts mid-sentence, and both speech and the brain stop. The audit log shows every
request the brain made, each one classified, with the spoken text traced to the wire,
and no speech generated from a request that was not a main turn.

## What the proxy reuses from cc-dump

The reusable part of cc-dump is `src/cc_dump/pipeline/`: `response_assembler.py`
(`ResponseAssembler`, `reconstruct_message_from_events`), `event_types.py` (the typed
event model, with `parse_sse_event` as the single SSE parsing boundary), and
`_extract_session_id` in `tui/stream_registry.py`, which reads the Claude Code session ID
from `metadata.user_id` (`user_<hash>_account_<uuid>_session_<uuid>`, present on every
request). The transport is rewritten using asyncio, because cc-dump's transport uses
threaded `http.server` and hands runs under Pipecat. The TUI is not needed.

## Not verified

- Whether a fork (a `/btw`-style side request) can be triggered from stdin in `-p` mode.
- How the stdin interrupt control request behaves with a slim `-p` brain, and how a
  partially completed turn is recorded in its history.
- The four timing intervals from the experiment; none can be measured from the existing
  recordings.
- How to distinguish a fork, including a compaction, from a main turn on the wire.
- Whether Claude Code's retry after a 529 error or a dropped stream is byte-identical to
  the original request.
- Which settings disable auto-memory, LSP, and background prefetches without `--bare`.
- Whether the ~300 ms requests are `count_tokens` calls: their body and response match
  that format, but the recorded URL is `/v1/messages`, so cc-dump may have normalized
  it.
