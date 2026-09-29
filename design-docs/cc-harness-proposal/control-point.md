# The proxy as the point of control

Written 2026-09-29 from a conversation between Brandon and Claude. This supersedes the
earlier documents in this directory, which describe the proxy as a tap that overhears.
It is not a tap. It is the point of control on the wire between the intermediary's
brain and the API, in both directions, and that changes what hands can be.

## The picture

```
[user] -> [mic] -> [hands: Pipecat pipeline] -> [brain: slim Claude Code] -> [working Claude Code sessions]
                          ^                              |
                          |    A: every model input       v
                          |       and output         [proxy: hands]  <->  api.anthropic.com
                          +--------------------------------+
```

Two Claude Code boxes, kept apart. The brain is one long-lived, slim Claude Code
process owned by the hands daemon: the intermediary's system prompt, a few built-in
tools, hands' session tools over MCP, skills as SKILL.md. It runs on the Claude
subscription because it is Claude Code making Claude Code's own requests. It never runs
in a project directory and never edits code. The working sessions are Brandon's ordinary
sessions, driven and observed as they are today. In Brandon's words: "not user -> mic ->
[single claude code session that is both hands AND the instance of claude code that
does the work]".

Pipecat stays whole. Mic, gate, VAD, Whisper, turn aggregation, TTS, speakers, barge-in
are untouched. Only the LLM stage of the pipeline changes: `build_llm` in
`src/hands/voice/pipeline.py` stops constructing an Anthropic or OpenAI service and
becomes a processor that hands the transcript to the brain and emits what comes back
over the wire as the frames TTS already consumes. Brandon: "reaching for the stars here
and not giving up all the magic that Pipecat provides."

## Two sources, one rule

There are two places hands can learn things about the brain:

- **A, the wire.** Every request the brain makes and every byte the API returns pass
  through the proxy, request before response. A is complete: nothing the model reads or
  writes exists outside it.
- **B, the harness.** Claude Code's stdout stream, its stderr, its exit code, and the
  hooks it fires. B carries what the harness did around the model: when a tool ran,
  what a permission decided, that the process died.

The rule: primary facts from A, derivative facts from B, no fact from both. Hands
speaks from A. B is reserved for what only the harness knows. Listening to A and B for
the same fact is not two streams, it is the same stream heard at two points in time,
and hearing it twice is the bug.

A is not merely earlier than B. A is complete and B is not: B shows what the harness
chose to surface, A shows everything the model was given and everything it said.

### What looked like problems from the tap's point of view

Each of these was a concern while the proxy was imagined as overhearing. Under control
they are policy.

**Retries.** The proxy sees the 529 or the dropped upstream stream before Claude Code
does. It passes the error through and recognises Claude Code's byte-identical
re-request, so hands knows which attempt is which. Retrying upstream itself would be a
request of the proxy's own on Claude Code's credentials, which it never makes (below).

**Compaction and utility calls.** The proxy sees the request before any response bytes
exist. A utility call has no tools and a different model. A compaction is a fork (below),
which shares the brain's prefix as a main turn does; what marks it is found by use.
Classification happens on the way out
and the response is already tagged when it arrives. This does not need to be exhaustive
up front. Brandon: "Simply using the application should reveal this quickly with no
effort." The one thing that must exist from the first line is the default: an
unrecognised request shape is never spoken, and is logged loudly.

**Refused or errored tools.** Also on A: the tool result, error flag included, comes
back in the next request's messages. B adds when the tool ran, not whether.

**Turn end.** A trivial signal from the harness: the stdout result event, a hook, the
exit code. There is no case where the harness disappears without a trace, because
disappearing is itself the trace.

## Owning the request

Because the proxy composes what the model sees, hands is no longer limited to what
Claude Code puts in the conversation. Prompt caching sets the rules for how far that
goes. (Caching mechanics below are documented API behaviour, not re-checked on the day
of writing.)

The cache is an exact-prefix match up to a breakpoint. A change at byte N invalidates
everything after N. So:

- **The stable body is byte-identical turn to turn.** Instruction, tool schemas,
  history. The proxy never rewrites the front or the middle per turn.
- **Per-turn material goes only at the tail.** Session status, a fresh note from hands,
  appended to or inside the newest user message. On the next turn Claude Code's history
  will not contain what the proxy added. Appended after the block that carries Claude
  Code's breakpoint on that message, the tail is outside the cached prefix and the next
  request misses nothing. Anywhere earlier, the rest of the message after it goes uncached
  once, and mid tool loop that message is a tool result of any size. It also means
  stale notes never accumulate in history the way `[hands]` messages do today. If zero
  miss is ever wanted, the proxy re-inserts its own past insertions, a pure function of
  its own log.
- **History is edited only in batches.** Trimming an old tool result invalidates the
  prefix from that point, so trims happen every K turns, all at once, one cache rebuild
  accepted. Compaction has the same shape for the same reason.
- **Claude Code already caches the way we'd want.** In the newest recording the system
  prompt carries `cache_control: {type: ephemeral, ttl: 1h}` and a third breakpoint sits
  on the latest user message. The proxy can move or add breakpoints if the tail ever
  needs one placed differently, but it does not have to.

The tail is the product surface. What hands writes into it each turn replaces today's
start-up note and `[hands]` messages, and it is composed fresh every turn from the
registry rather than accumulated.

## Who has the intelligence to manage the brain's own context

The brain cannot summarise itself while it is answering, the proxy makes no requests of
its own, and the rule against reusing Claude Code's OAuth token from another client
stands. Three answers, in the order to reach for them:

**Trimming by rule, no model.** The brain's context has a known shape: short spoken
turns, ephemeral notes in the tail, and tool results, which are where the volume is. A
`read_session` result from twenty minutes ago becomes a one-line stub by rule, in the
pure core, batched every K turns.

**Forks, which Claude Code already has.** `/btw` is implemented in
`src/utils/forkedAgent.ts` of the research fork as a forked agent: a second query loop
that reuses the main loop's exact system prompt, context, tools, and messages (the
"cache-safe params, must match parent for cache hits"), appends a side question, and
never writes its messages into the main conversation. The same mechanism runs
`postTurnSummary`, `promptSuggestion`, `session_memory`, and `compact`. This is the
side request we thought the brain lacked: subscription-legit, sharing the brain's prompt
cache so it costs only the tail and the output, and leaving no trace in history. On the
wire a fork shares the brain's prefix, as a main turn does; what tells the two apart is
not verified. `/btw` itself is an interactive command; whether a fork can be triggered
from stdin in `-p` mode is not verified. If it cannot, exposing one is one of the
patched build's two jobs (see "What needs Claude Code modified").

**Steering compaction.** Claude Code's own compaction is a fork that flows through the
proxy. Rather than switching it off, the proxy recognises it on the way out, rewrites
its summarisation prompt to what a voice session needs kept (session titles, the draft
in flight, decisions taken), and lets the response replace Claude Code's history. One
cache rebuild, on our terms.

The local model hands already has is held in reserve for summarisation off the
subscription. It is not designed around.

## The summary store

Brandon's question: is a cache of summaries over individual pieces of content, indexed
and invalidated by hash, worth building, or are reads so cheap it is more effort than it
is worth? The links backlog answered it. Without reading any of it:

- 132 open tickets. `lit show` over all of them: 7,149 lines, 579 KB, about 145K tokens.
- Raw descriptions alone: 184 KB, about 46K tokens, median 1.3 KB per ticket.
- The whole tracker with 634 closed tickets: 1.4 MB, about 358K tokens.
- The listing by itself: 11 KB, about 2.8K tokens.

"What's in the backlog?" answered by reading it raw is a context the brain cannot hold,
and where it could, tens of seconds of uncached reading per question. It is large,
mutable, and re-read constantly. So the store is worth building, in this shape:

- **Keyed by content hash** of title plus description, with the summariser's version in
  the key. Not by `updated_at`: a rerank touches that without changing what the ticket
  says and should not cost a re-summary.
- **One sentence per ticket.** 132 × ~25 tokens is about 3K tokens: the whole backlog
  fits in a spoken-scale context, and the raw ticket is one call away.
- **Layered.** A parent's summary is keyed by the hash of its children's summaries, up to
  the backlog, whose children are its epics and the tickets under no epic. Edit one ticket
  and only it and what sits above it recompute. "How's
  the observability epic going?" becomes a lookup.
- **Summarised off the voice path.** On first sight and on change, never while the user
  waits, by a fork of the brain or by the local model.
- **One table for everything content-addressed.** Files are by nature; a finished turn
  of a work session is immutable and keyed by its id. Per-turn summaries hands already
  produces for narration go in the same store, so `read_session` serves summaries first
  and raw content on request: a few hundred tokens in the brain's context instead of
  thousands.

This is also the substrate the endurance epic (`hands-memory-5qk`) wanted. Recall is a
search over summaries that already exist, append-only where the content is immutable.

Concrete effect in hands: a `summaries` store, and `read_backlog`, `read_ticket`, and
`read_session` tools that push pointers and pull content, the shape
`docs/architecture.md` already names.

On lit's `prompt` field, in Brandon's words: "an idea that essentially never quite became
necessary. The entire output of every command is a prompt, in essence."

## What Claude Code adds beyond what you typed

From the newest recording (`ccdump-20260422-233238Z-2be307e3.har`, Claude Code
2.1.111), so the slim brain knows what it is removing:

- System prompt: three blocks, 81 + 57 + 27,574 characters, the last two with 1h cache
  breakpoints. Body fields beyond the obvious: `thinking: adaptive`,
  `context_management` clearing old thinking, `output_config`, `metadata.user_id`
  carrying device, account, and session ids.
- First user message: 509 characters typed, 47,889 injected as `<system-reminder>`
  blocks: the skills list (15K), CLAUDE.md and context (30K), deferred tools (2K),
  SessionStart hook output (1K).
- Later turns: 0–1.1K characters injected on a typical turn; 45K on a turn carrying a
  Read result with plan-mode reminders. Fifty-seven tool results over the session
  totalled 171K characters.
- The ~300 ms requests have the shape of `count_tokens` calls: no `max_tokens`, a
  file's contents as the one message, and `{"input_tokens": N}` back. They size file
  reads before adding them to context. Not model calls, but round trips, in bursts of
  four.

## What needs Claude Code modified

Less than expected, but not nothing. Read from `claude --help` on 2026-09-28:
`--system-prompt` replaces the prompt; `--tools` names the built-in set, `""` disables
it; `--strict-mcp-config` with `--mcp-config` loads only hands' server;
`--setting-sources` narrows what is read; `-p --input-format stream-json
--output-format stream-json --include-partial-messages` is the transport;
`CLAUDE_CONFIG_DIR` isolates the brain's config, skills, and login; `ANTHROPIC_BASE_URL`
routes it through the proxy, as cc-dump proves daily. Interrupt over stdin is a control
request the CLI's stream-json protocol carries for its own SDK; that it works against a
slim `-p` brain is listed below as unverified.

`--bare` is out. Brandon knew this from use; the help text and the source confirm it:
"Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings (OAuth and
keychain are never read)", and `auth.ts:101` in the research fork: "--bare:
API-key-only, never OAuth." The `CLAUDE_CODE_SIMPLE` variable is the same switch
(`isBareMode()` in `envUtils.ts:60` is the variable or the flag), so there is no back
door. Simple mode would also cut more than wanted: tools become Bash, Read, and Edit
only, and the system prompt a three-liner. What `--bare` would have skipped therefore
needs its own handling: hooks and plugin sync by a config directory with none; CLAUDE.md
discovery by a working directory with none; auto-memory, LSP, and background
prefetches by their own settings, to be found.

So the patched build now has two concrete jobs, each small: decouple OAuth from bare
mode in `auth.ts`, about four lines, if the per-feature settings prove insufficient; and
expose a fork headless, if one cannot be driven from stdin. Everything else is flags.
And for any utility request that survives, the no-patch option remains: the proxy
answers it itself and never goes upstream, which cc-dump's `sentinel.py` already does
for `$$` messages. That rewrites a response to the client, not a request to Anthropic.

## Timing is measured by the proxy, not by an experiment

An earlier draft put a timing experiment first. It is not a gate, and it is not a
ticket. Brandon's position on latency, which the recordings support: the same bytes on
the wire take the same time whether Claude Code or anything else sent them; a
short-output Opus request with a 27 KB prompt, 67 tools, and 100–200K cached tokens
returns in 2.3–3.5 s. The recordings also
already answer how many requests a turn makes: more than one, and the section above
says which. What they cannot supply is timing around the request, and no value of it
changes the design:

- stdin write → the request leaves Claude Code: if large, a setting or a patch in the
  brain's launch, not a change of shape;
- request leaves → first SSE byte: the API's floor, the same in any design;
- first SSE byte → the same text on Claude Code's stdout: would only matter if it
  decided whether hands reads the wire or stdout, and that is decided on completeness;
- requests per turn: handled by the classification default and discovered by use.

So the proxy timestamps every request on the wire from its first version (request in,
request out, first byte, last byte), and those numbers fall out the first time the brain
answers a question. The stdin and stdout ends are not on the wire; the brain's launcher
sees them. The measurement happens last, by construction, as the acceptance check of the
brain launch.

## Implementation plan

Filed in lit on 2026-09-29 as epic `hands-wire-6ic` with the tickets below as its
children, in this order. Another session can recreate the backlog from this section
alone; each ticket points here by path. Question tickets sit ahead of the work they gate, so a builder never stops to
investigate mid-build.

1. **The proxy.** An asyncio reverse proxy in hands that Claude Code reaches through
   `ANTHROPIC_BASE_URL`: forwards each request upstream unchanged, streams the response
   back byte for byte, and feeds the same bytes to an assembler ported from cc-dump's
   `response_assembler.py` and `event_types.py`. Every request is attributed to a
   session by `metadata.user_id`, classified on the way out (main turn, fork,
   compaction, count_tokens, unknown), timestamped at request-in, request-out,
   first-byte, and last-byte, and written to the audit log. Unknown shapes are never
   spoken and are logged loudly. The proxy never makes a request of its own and never
   reuses the client's credentials. Purpose in one sentence: everything one Claude Code
   process says to the API and hears back, as typed events. Second consumer: the
   working sessions (ticket 10).
2. **Question: which settings replace `--bare`.** `--bare` and `CLAUDE_CODE_SIMPLE`
   both disable OAuth. Find the settings or environment that switch off auto-memory,
   LSP, background prefetches, and CLAUDE.md discovery for a `-p` process while OAuth
   stays on; if none exists for some feature, say which, so the brain-launch ticket
   knows whether the `auth.ts` patch is needed. Answered by sharpening ticket 4.
3. **Question: interrupt over stdin.** In `-p --input-format stream-json` mode, send the
   interrupt control request mid-turn and record what Claude Code does: whether
   generation stops, what lands in the session history, what stdout emits. Answered by
   sharpening ticket 5.
4. **The brain launch.** A `HANDS_LLM` variant in `src/hands/daemon/run.py` that starts
   one long-lived slim Claude Code: `CLAUDE_CONFIG_DIR` at a hands-owned directory,
   `--system-prompt` from `src/hands/voice/intermediary_instruction.py`, `--tools` to
   the built-in set decided there, `--strict-mcp-config` with hands' MCP server,
   `-p --input-format stream-json --output-format stream-json
   --include-partial-messages`, `ANTHROPIC_BASE_URL` at the proxy, and the settings
   ticket 2 found. The MCP server is a second adapter over the tool bodies in
   `src/hands/voice/tools.py`, which stop being Pipecat `DirectFunction`s and become
   plain callables with schemas; the Pipecat shell stays as the first adapter and the
   `audited` wrapper stays on the bodies. Done when the brain answers a typed question
   over stdin with its text visible on the proxy, and the four timestamps for that
   turn are in the audit log.
5. **The LLM stage.** `build_llm` in `src/hands/voice/pipeline.py` gains a processor
   that sends the aggregated transcript to the brain's stdin and emits the wire's text
   deltas as the frames TTS consumes today. Barge-in forwards the interrupt (per ticket
   3). `stay_silent` ends the turn with nothing said after it, as `run_llm=False` does
   today; the draft tools complete their
   effect regardless of interrupt and their readback is spoken from the outcome. Done
   when a person asks by voice and hears the answer, and the audit log shows the text
   came from the wire.
6. **The tail.** What hands appends to the newest user message each turn, composed
   fresh from the registry: session status, and what the start-up note and `[hands]`
   messages carry today. Stable body untouched; one turn's tokens uncached per turn is
   the accepted cost.
7. **The summary store.** A store keyed by content hash plus summariser version, one
   sentence per item, layered so a parent's key is the hash of its children's
   summaries. `read_backlog`, `read_ticket`, and `read_session` tools serve summaries
   first and raw content on request. Summarisation runs off the voice path. Purpose:
   a sentence for any content-addressed thing, kept until the thing changes. Second
   consumer: narration's per-turn summaries, and lit itself if the summaries are
   written back. Numbers that size it: 132 open tickets, 145K tokens via `lit show`.
8. **Question: can a fork be driven headless.** `/btw` runs a forked agent
   (`src/utils/forkedAgent.ts` in the research fork) that shares the brain's prompt
   cache and leaves no trace in history. Determine whether a `-p` process can be asked
   for one from stdin. Answered by sharpening ticket 9.
9. **Forks and compaction steering.** Far: the brain's context managed by rule-trimming
   in batches, forks for summaries, and Claude Code's compaction recognised on the way
   out and its prompt rewritten. Not pinnable until ticket 8 answers whether forks are
   reachable; if they are not, the patched build's second job is to expose one.
10. **Work sessions through the proxy.** Fritter's `claude` shim sets
    `ANTHROPIC_BASE_URL` (beside `FRITTER_SOCKET`, `src/hands/sessions/shim.py:90`) so
    every wrapped session flows through the proxy, and narration gets the model's
    stream instead of only Stop hooks. Observation only; separable.

The epic's checkpoint, in vivo: a person asks by voice what a session is doing, what is
in the backlog, and to file a ticket; hears each answer; barges in mid-sentence and
both speech and the brain stop; and the audit log shows every request the brain made,
each classified, with the spoken text traced to the wire and nothing spoken from a
request that was not a main turn.

## What the proxy borrows from cc-dump

cc-dump's `src/cc_dump/pipeline/` is the reusable part: `response_assembler.py`
(`ResponseAssembler`, `reconstruct_message_from_events`), `event_types.py` (the typed
event model, `parse_sse_event` as the one SSE boundary), and
`tui/stream_registry.py`'s `_extract_session_id`, which reads the Claude Code session id
out of `metadata.user_id` (`user_<hash>_account_<uuid>_session_<uuid>`, present on every
request). The transport is rewritten on asyncio, since cc-dump's is threaded
`http.server` and hands runs under Pipecat. The TUI is not needed.

## Not verified

- Whether a fork (`/btw`-style side request) can be triggered from stdin in `-p` mode.
- The stdin interrupt control request against a slim `-p` brain, and how a half-finished
  turn lands in its history.
- The four experiment intervals; none can be read from the existing recordings.
- What tells a fork, compaction included, from a main turn on the wire.
- Which settings switch off auto-memory, LSP, and background prefetches without
  `--bare`.
- That the ~300 ms requests are `count_tokens` calls: their body and response have that
  shape, but the recorded URL is `/v1/messages`, so cc-dump may have normalised it.
