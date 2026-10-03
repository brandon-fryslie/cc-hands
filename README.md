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
  is right, amends or discards it, and sends it by typing it into the session through
  fritter, the pty wrapper in `fritter/`.
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
- [fritter/README.md](fritter/README.md): the pty wrapper that types into a session,
  its protocol, and what was measured against Claude Code.
- [docs/failure-modes.md](docs/failure-modes.md): what goes wrong, observed and
  anticipated, and the rule each entry produced.
- [docs/happy-voice-reference.md](docs/happy-voice-reference.md): how Happy, the
  closest prior art, works, and what to copy and avoid.

## Shape

One Python process, `hands`, run in a terminal, with a Pipecat voice pipeline on one
side and the Claude Code plumbing on the other, meeting through a pure core.

```
mic ──► gate ──► Whisper (MLX) ──► LLM ──► pocket-tts ──► speakers
                                    │   ▲
                         tool calls │   │ hook events, as frames
                                    ▼   │
                             sessions + core
                     registry · drafts · JSONL reader · audit log
                                    │
                     hook replies · fritter sockets
                                    ▼
                     target Claude Code sessions (any terminal, started any way)
                                    │
                  hook shims ──► unix socket ──► sessions
```

hands' settings are `~/.hands/config.toml`; with no file, every setting is its default. Save an
edit while hands runs and it restarts on it, keeping its sessions; an edit it cannot read, or one
naming a backend whose key or login it lacks, is said in `hands log`, and hands runs on as it was:

```toml
[llm]
backend = "claude"            # "anthropic" (the default), "openai", or "claude"
model = "claude-sonnet-5"     # another model, for any backend
url = "https://..."           # another server, for anthropic and openai

[whisper]
model = "mlx-community/whisper-large-v3-turbo"

[telemetry]
collector = "http://otel.example:4318"   # an OpenTelemetry collector's OTLP/HTTP address
```

With a collector set, each wide event hands writes to its audit log is also sent there as a
span, so it can be found in whatever stores and Grafana sit behind the collector. The log stays
the whole record either way; a batch the collector did not take is an `Undelivered` line in it.

The LLM is a backend variant: `anthropic`, the default, is Claude Sonnet 5, keyed by
`ANTHROPIC_API_KEY` or, when that is not set, by the keychain's `HANDS_LLM_ANT_KEY`; `openai`
is `gpt-4.1-mini` through OpenAI's API, keyed by `OPENAI_API_KEY`. `model` names another
model for either. `url` moves either to another server that speaks its API, keyed only by
the environment's key (the keychain's key is Anthropic's own and goes nowhere else), and the two take
it differently: the Anthropic client appends `/v1/messages`, so its URL has no `/v1`
(`https://api-chicago.codexapi.pro`), while the OpenAI client appends `/chat/completions`, so its URL
usually ends in `/v1` (`https://api-chicago.codexapi.pro/v1`: the bare host answers 404). The server
must stream tool calls, because the pipeline's service always streams: api-chicago.codexapi.pro streams
Anthropic `tool_use` but drops OpenAI-shape tool calls (2026-09-25), so it is reached as `anthropic`.
`claude` is the brain: one long-lived Claude Code of hands' own, on the Claude subscription,
whose requests go through hands' proxy and which reaches the sessions through hands' tools over MCP
(`src/hands/brain/`). It takes no key; its login lives in `~/.hands/brain`, set up once as any
Claude Code is, by running `mkdir -p ~/.hands/brain/cwd && cd ~/.hands/brain/cwd && CLAUDE_CONFIG_DIR=~/.hands/brain claude` and answering its first screens, and it does not start without one. `hands login` logs it in again, or onto
another account, and says which account it holds after.
Its tools, its permission rules and mode, and its MCP servers are that directory's, as for any Claude Code:
`~/.hands/brain/settings.json`, and `CLAUDE_CONFIG_DIR=~/.hands/brain claude mcp add -s user ...`; hands adds only its
own MCP server, which it may use without asking, and its hooks, and it keeps the account's claude.ai connectors out.
Nobody sits at its keyboard to answer a dialog. A permission its settings ask about is put to the user by voice, in
the turn of theirs that led to it: hands says what the tool would do, and only a plain yes, said once the question has
played to its end, runs it. Anything else they say refuses it and is handed to the brain, and a permission nobody can
be asked, or nobody answers in time, is refused; a `BrainPermission` audit line records how each was settled. Any
other dialog is refused: the brain asks its questions in its reply, and an MCP server asking for input gets a
`BrainRefused` audit line. Its settings say `"defaultMode": "default"` with the tools it may use without asking in
`permissions.allow`, and `"syncClaudeAiSkills": false` and `"syncClaudeAiPlugins": false` keep the
account's skills and plugins out, which Claude Code reads from no other place. A `BrainOffered` audit line names the tools
its requests offer, at its first request and whenever they change.
It works a project's backlog itself, with `lit` from its shell in the project's repository, as `lit quickstart` there
says; `read_backlog` names where in it to run `lit`.
It searches the web and reads pages with the `firecrawl` command from its shell, on that command's own login, as a
skill of its setup says: `~/.hands/brain/skills/firecrawl/SKILL.md`, with `"deny": ["WebSearch", "WebFetch"]` in its
settings so Firecrawl is its web. As a skill it adds one line to each request until it is used (about 110 bytes,
2026-10-03); Firecrawl's MCP server would add every one of its tools' descriptions to every request.
Its prompt names its setup's directory, so a skill you ask it by voice to install, change, or remove is a folder in
`~/.hands/brain/skills`, never in your `~/.claude`; it has a new skill from its next turn, with no restart.
Beside them, hands gives it skills of hands' own, shipped with the code they serve (`src/hands/brain/plugin`):
`hands:prompt` is how it writes a draft for a session from what you said, a small cut of `laws:prompt` made for that one job.
`hands:chat` is how it talks with you, a small cut of `laws:chat` made for replies you hear.
It is interactive Claude Code under hands' own fritter, never `claude -p`: hands types each turn into its input
and stops a turn with Escape, and types nothing else into it; its hooks say when a turn was taken and when it ended.
What hands asks in the background, the Claude Code backend's summaries and the sentence an old tool result goes as,
is asked of a second interactive Claude Code started for each question, with `/btw` and the question as its opening
prompt (`src/hands/brain/asides.py`), so a spoken turn never waits on one; each has its `AsideAnswered` audit line. It is the pipeline's LLM stage (`src/hands/brain/stage.py`): what
it says is spoken from its requests on the wire, never from its screen; each turn's `BrainSpoke` audit line names the
exchanges its words came from. A barge-in stops it, except while a tool whose effect must land is running,
which finishes and has its readback, or why it failed, spoken; after that, and after `stay_silent`, hands
answers the brain's next request itself, so the model is not asked to go on. A turn the brain ends in error is
said as the other variants' model failures are.
A variant stops at start, naming where its key can be, when it has none. The run's audit log says which
backend, URL, and model it reached, never the key. The key can live in a `.env` at the repository root,
which git ignores, and `uv run --env-file .env` puts it in the environment; uv stops if
the file is not there. The gate is push-to-talk: the key is the voice activity detector and the microphone mute,
so the turn boundary is the key and the pipeline can never transcribe itself. Measured
on 2026-09-12, voice to voice with a local Qwen3-30B-A3B since retired: 1.4 s from key release to first
audio on a plain turn, 4.3 s on a turn with a tool call.

## Installing the hooks

hands hears a session through Claude Code hooks, and they come from a plugin. This
repository is a marketplace holding that plugin, `plugin/`, so installing it installs
the hooks in every session, and nothing is merged into your settings by hand:

```
claude plugin marketplace add ~/code/cc-hands     # once; in a session: /plugin marketplace add ~/code/cc-hands
claude plugin install hands@cc-hands              # hooks on, in every new session
claude plugin disable hands@cc-hands              # hooks off, still installed
claude plugin enable hands@cc-hands               # hooks back on
claude plugin uninstall hands@cc-hands            # hooks gone
claude plugin marketplace remove cc-hands         # and the marketplace with them
```

The same commands work as `/plugin ...` inside a session. A session picks up a change
when it starts, or on `/reload-plugins`. Installed from a local directory, the plugin
runs from this checkout, so a `git pull` here updates the hooks too.

The plugin also holds two skills. `/hands:restart` restarts a running hands from the
session you are in, so it runs the code, the brain's prompt, and the brain's setup on
disk now: hands stops as `q` stops it and starts again in the same terminal and process,
with the same menu-bar item and every running session still listed, and the skill says
so in one line once the new run is up, usually within seconds. No change is taken up
without a restart; a skill added to the plugin reaches a session on `/reload-plugins`.

`/hands:attention`, which you can also say to hands in your own words ("stop telling
me when sessions finish", "be quiet for a while"), sets what hands says without being
asked: each finished turn (`finished off|brief|full`), the focused session's steps as it
works (`progress off|brief|full`), and a session ending (`ended on|off`). `quiet on`
holds all of it until `quiet off`, and leaves the rest as it was set. With no
arguments it says what is set. Finished turns are off until you turn them on; while they
are off, only the turns of a session you asked hands to watch are told as they finish,
and any session's last turn is told when you ask for it. Permission requests, questions
asked in a dialog, and plans are spoken whatever is set. Settings last across restarts
and take effect from the next thing hands would have said.

Hands speaks in Charles, one of Pocket TTS's own voices, until you choose another. Ask
it which voices there are and to let you hear some: each says a line in its own voice.
Tell it which one to use, and what it says next is in that voice. The choice is kept in
`~/.hands/voice`, so it lasts across restarts.

The hooks need a Python 3.12 or newer on `PATH` (`python3.14`, `python3.13`,
`python3.12`, or a `python3` that is new enough); they run hands' own `src` and need no
venv. Without one, every hook fails saying so. The shim finds hands' home as the CLI
does: `HANDS_HOME`, which must be absolute, or `~/.hands`.

## Wrapping every session

hands types into a session through fritter, so it reaches only the sessions started
under it. Install a `claude` that starts every one that way, and put it first on `PATH`:

```
uv run hands install-fritter                    # builds fritter, writes ~/.hands/bin/claude beside it
export PATH="$HOME/.hands/bin:$PATH"            # in your shell's startup file
```

That `claude` runs the next `claude` on `PATH` under fritter when a terminal is on both
ends and there is no `-p` or `--print`; a pipe, a script, and `claude -p` run the real
claude exactly as before. `hands install-fritter` exits 0 only when `claude` on the
current `PATH` is the one it wrote, and says what to add when it is not. Run it again
after fritter changes: it builds fritter from this checkout. A session started before
the shim stays unreachable until it ends.

A session under fritter reaches the API through its own fritter, which is the session's
proxy: the session still names the API it would have used, fritter answers its
connections to that API's host with a certificate of its own that only the session is
given to trust, forwards each request on, and sends hands a copy of the exchange, so the
audit log holds every request each session makes, under its session id. Every other
connection goes through fritter unopened. Because the session still names Anthropic's
API, Claude Code keeps all it keeps for it, Remote Control included. The session never
waits on hands: with hands stopped, its requests go through as before, and the copies it
could not hand over are counted in the next one hands takes. The session's shell
commands, and any `claude` started from it, run with the proxy the session had before
fritter. A firewall that asks per program, such as Little Snitch, sees the session's
connections as fritter's and holds each one until it is answered: allow
`~/.hands/bin/fritter` out, and again after `install-fritter` rebuilds it.

The hooks are on whether or not hands is running. While hands is stopped, has never
run, or is still starting, they cost a session nothing: no hook error, and a permission
request gets Claude Code's own dialog. A hands that died, hung, or left a heartbeat nothing can read shows
up in every session as a hook error saying so.

`plugin/hooks/hooks.json` is generated from `hands.sessions.hookconfig`, and a test fails when
the two differ. After changing the hook table:

```
uv run python -m hands.sessions.hookconfig > plugin/hooks/hooks.json
```

## Running

```
uv sync
uv run hands run                        # the backend ~/.hands/config.toml names; hold Right Shift in any app to talk, release to send; q in its terminal quits
uv run --env-file .env hands run        # its key in .env: ANTHROPIC_API_KEY (else the keychain's HANDS_LLM_ANT_KEY) or OPENAI_API_KEY
uv run hands status                     # up, stopped, not responding, down, or never ran; exits 0 only when up
uv run hands check                      # whether hands is set up to work here; exits 0 only when every piece is
uv run hands log                        # the audit log: what hands heard, said, called, and failed at
uv run hands phone                      # the addresses a phone opens the talk page at, the first as a QR code; served while `hands run` is up
uv run hands indicator                  # the daemon's verdict in the menu bar; `hands run` starts one
make check                              # pytest, pyright, and fritter's Go tests; fails when any of them fails
uv run python evals/intermediary.py    # conversations through the intermediary's prompt and tools; needs the model to be up
```

`hands check` looks at each piece hands needs and says it is there or what puts it
there: the plugin, installed and enabled; `claude` on this `PATH` being hands' shim; this
terminal's Input Monitoring grant; each running session hands knows of that cannot be
typed into; and each running session hands has no record of, such as one started before
the plugin was installed; both by its directory and pid. `hands run` says the same lines as it starts. An
up daemon is not a working hands: `hands status` says only whether the daemon is running.

`hands run` needs the Input Monitoring grant for the terminal app it runs in (System Settings > Privacy &
Security > Input Monitoring) to hear Right Shift from other apps; without it, it names the grant and exits.

`pytest` and `pyright` judge the code. The eval judges what a listener hears: it tells four
real turns, lifted whole out of real transcripts, and checks that the facts are there, that
no code name reached the ear, that what is spoken stays within its configured length, and
that every number the model said is a number the turn showed. It exits 0 when every check held, 1 when one
failed, and 2 when the model could not be reached at all.

The intermediary's eval judges its next move at one moment of a conversation: the tool it
calls and with what, or what it says, held to the same spoken-form judge, or that it stays
silent for words not meant for it. Its exit codes mean the same.

hands runs only while `hands run` does, in the terminal it was started in: `q` or
Ctrl-C stops it, and nothing starts it again.

`hands run` starts the menu-bar indicator beside it. The indicator's title is ✋ while
the daemon is up. It reads "hands stuck", "hands down", or "hands unreadable" when
something is wrong, and "✋ off" when the daemon was stopped or never ran. It posts a
notification when the daemon stops being up, and once the daemon is gone it posts that
and goes away too.

Every heartbeat rewrites `~/.hands/status.json`, every effect and failure is a line
in the segmented log `~/.hands/audit/` (two 32 MiB segments at most), and the daemon's output goes to its terminal. Each line's
`level` is `error` for anything that went wrong, so `jq -cR 'fromjson? | select(.level == "error")'`
finds them all; the brain is told the same and reads the log itself when asked what happened.

## Prior art

Happy (`~/code/happy`) does hands-free Claude Code over ElevenLabs ConvAI. Its core
workflow works and is why this approach is worth building; its trouble was transport
reliability, not content. Its two ideas worth keeping are the silent-context versus
speaks-now channel split and the read watermark. The full reading is in
[docs/happy-voice-reference.md](docs/happy-voice-reference.md).
