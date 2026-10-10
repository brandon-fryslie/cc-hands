# The hands guide

This guide covers everything after [getting started](../README.md): what hands does, its commands and settings, the brain, the plugin, and how hands connects to sessions.

## What hands does

The brain, hands' intermediary, makes the workflow
eyes-free as well as hands-free: a full voice-to-voice loop that requires neither the
user's hands nor eyes. The workflow has been tested and works well in practice.

A basic hands-free workflow involves proofreading STT output and re-reading the original
when TTS output is garbled by special characters. The intermediary handles both. To
proofread a message before it goes to Claude, have the agent read it back. You can
explore Claude's output interactively, on demand. The intermediary is designed to be
fast.

Follow-up questions work the same way: you hear a summary and ask "give me the details of
that part." A single stateful agent handles both summarization and Q&A, so the agent that
produced the summary still has the content it summarized.

When complete, hands does the following:

- Takes audio and writes the prompt for the coding session. Reads a draft back until it
  is correct, amends or discards it, and sends it by typing it into the session through
  fritter, the pty wrapper in `fritter/`.
- Lists the running sessions, reads what a session has done, starts and ends sessions,
  and switches the session you are talking to.
- Answers permission prompts, questions, and plan approvals by voice. If you do not
  answer, it denies by default.
- Interrupts a session and runs the session's slash commands.
- Reads a summary of the session's response aloud. On request, it reads the details, the
  exact text, or the test output. It also answers follow-up questions about anything it
  said.

## Checking an install

`hands check` checks each step of the [install](../README.md#install) and reports whether
it is complete. For an incomplete step, it shows the command that completes it. It checks
the following: the native `claude` is on this `PATH`; PortAudio is installed; the `hands`
on this `PATH` is this hands, because Claude Code runs it for the plugin; the `claude` on
this `PATH` is hands' shim; the plugin is installed and enabled; Claude Code is logged in
and has no remaining first-run prompts in the smoke test's folder; the brain is logged in;
the app that hands runs in has the Input Monitoring permission; and hands is running. It
also reports each running session that hands knows about but cannot type into, through
its fritter or the tmux pane it runs in, and each running session that hands has no
record of, such as one started before the plugin was installed. It identifies both kinds
of session by directory and pid. `hands run` prints the same lines as it starts, and
prints the brain's line when it connects to the brain. A running daemon does not mean
hands is working. `hands status` only reports whether the daemon is running.

The smoke test runs in a separate terminal while hands is running. It calls hands the
same way the phone's page does and runs three spoken turns using the macOS voice: it asks
hands to have a session that it starts in `~/.hands/smoke` name the single file in that
folder, sends the draft, and asks what the session replied. Each stage of the pipeline
prints an `ok` line. The run stops at the first stage that fails and reports which stage
it was. It exits with code 0 only if every stage succeeds. The test runs a model turn in
that session, using your Claude Code login.

```
hands smoke
```

## Commands

In a checkout, run every command through uv. For an installed hands, use the same command
without `uv run`.

```
uv sync
uv run hands run                        # uses the brain and the model named in ~/.hands/config.toml; hold Right Shift in any app to talk, release to send; press q in its terminal to quit
uv run hands run --model claude-opus-5-5  # uses that model instead of the one named in config.toml, until it quits
uv run hands status                     # reports up, stopped, refused to start (and why), not responding, down, or never ran; exits 0 only when up
uv run hands check                      # checks whether hands is set up to work here; exits 0 only when every check passes
uv run hands restart                    # restarts the running hands with what is currently on disk, as /hands:restart does; exits 0 once it is running again
uv run hands log                        # shows the audit log: what hands heard, said, called, and failed at
uv run hands recall token helper        # searches what was said, sent to a session, and answered there, for entries that contain every word; the brain uses it to recall
uv run hands phone                      # prints the addresses where a phone can open the talk page, with the first as a QR code; served while `hands run` is running
                                        # the same address with /conversation before the # opens the conversation page: what was said and called, and a box for typing to hands
uv run hands indicator                  # shows the daemon's status in the menu bar; `hands run` starts one
uv run hands tmux-status                # prints the menu bar title for a tmux status line, coloured by status
make check                              # pytest, pyright, fritter's Go tests, and a type check of hands.app's launcher; fails when any of them fails
make app                                # builds build/hands.app, which runs `hands run` from the user's login shell and holds hands' Microphone and Input Monitoring permissions itself; signed with this Mac's Developer ID; logs to ~/Library/Logs/hands/hands.log; HANDS_POLAR_ORGANIZATION and HANDS_POLAR_PORTAL set the Polar organization that sells it; HANDS_POLAR_API=https://sandbox-api.polar.sh builds it against Polar's sandbox
make notarized-app                      # builds the same app, notarized by Apple and stapled, so Gatekeeper opens it on any Mac; HANDS_NOTARY_PROFILE sets the notarytool keychain profile
```

hands runs only while `hands run` is running, in the terminal where it was started.
Pressing `q` or Ctrl-C stops it, and nothing restarts it automatically.

To detect Right Shift in other apps, `hands run` needs the Input Monitoring permission for
the app it runs in: hands.app or the terminal app. hands.app asks for it in its setup window
(see "The packaged app"). For a terminal, run `hands grant` in that terminal, or grant it in
System Settings > Privacy & Security > Input Monitoring.
Without the permission, `hands run` reports the missing permission and exits.

## Settings

hands' settings are stored in `~/.hands/config.toml`. If the file does not exist, every
setting uses its default. If you save an edit while hands is running, hands restarts with
the new settings and keeps its sessions. If hands cannot read the edit, the edit was made
while the brain has no login, or the edit names a Claude model that hands does not offer,
hands reports the problem in `hands log` and continues running with its previous settings.
You can also ask hands by voice to change its own model ("switch yourself to Opus"). hands
confirms that it is switching, writes `model` into this file, and restarts with the new
model. `hands run --model claude-opus-5-5` runs on that model instead of the file's model,
until you quit it. While it does, hands refuses a model chosen by voice, because changing
the file's model would have no effect:

```toml
[llm]
backend = "claude"            # the brain: the default and the only option
model = "claude-opus-5-5"     # one of the four models below

[telemetry]
collector = "http://otel.example:4318"   # an OpenTelemetry collector's OTLP/HTTP address

[talk]
personality = "Dry and wry, with a bit of wit."   # how hands comes across, in your own words
wake_word = "Hey Mycroft"     # the phrase the wake word trigger listens for
```

`personality` changes only hands' tone and choice of words. Replies stay short and spoken,
and hands does only what you tell it to. If you omit it, hands uses its default plain style.

`wake_word` must be one of openWakeWord's built-in phrases: Hey Jarvis (the default), Hey
Mycroft, Hey Rhasspy, or Alexa. For any other phrase, train a model on it with
[openWakeWord](https://github.com/dscripka/openWakeWord#training-new-models), export it as
ONNX, and set its path next to the phrase: `wake_word_model = "~/wake/hey_computer.onnx"`.

hands can play, pause, and skip tracks in Spotify on this Mac, set its volume, shuffle, and
repeat, and report what is playing. It does this through Spotify's AppleScript interface
(`src/hands/spotify.py`). macOS asks once whether hands may control Spotify. Only the play
command starts Spotify. To play something you name, hands searches Spotify's catalogue.
This requires a Spotify developer app's credentials
([developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)) in hands'
environment as `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`. Without them, hands
refuses the search and reports why.

When a collector is set, each wide event that `hands run` writes to its audit log is also
sent to the collector as a span. Spans go directly to that address, bypassing any proxy,
so they can be found in whatever storage and Grafana are behind the collector. The log
remains the complete record either way. Each batch sent is recorded in the log as an
`Exported` line that lists its spans and, if the collector did not accept them, the reason.

## The brain

hands communicates through a harness it controls, never through a model API called with a
key. That harness is `claude`, the brain: a single long-lived Claude Code instance that
belongs to hands. It works with any login that Claude Code accepts from Anthropic, and it
uses Claude Sonnet 5.5 unless `model` names another of the four models hands offers:
`claude-haiku-4-5-20251001`, `claude-sonnet-5-5`, `claude-opus-5-5`, or
`claude-fable-5-1`. Its requests go through hands' proxy, and it reaches the sessions
through hands' tools over MCP (`src/hands/brain/`). It does not use a key. Its login is
stored in `~/.hands/brain`, and it does not start without one.
A restart resumes the brain's conversation instead of starting a new one. To start a new
conversation, remove `~/.hands/brain/conversation` before starting hands.
`hands login` sets up the brain. It writes the brain's initial `settings.json` if none
exists, and never changes an existing one. hands answers the brain's first-run screens
that only set preferences (the theme, and trust of the directory the brain runs in)
itself, in `~/.hands/brain/.claude.json`. As a result, on a brain with no login,
`hands login` runs only Claude Code's own login, which opens your browser. The one
question hands cannot answer is whether to use an API key set in the brain's
`settings.json`. In that case, Claude Code's first-run setup still runs; you answer the
question there and enter `/exit`. The brain logs in with a Claude plan and does not ask
which type of login to use. `hands login --console` uses an Anthropic Console key
instead. If the brain has completed setup and has a login, `hands login` asks nothing.
`hands login --claudeai` logs it in again, or into another account, with a Claude plan,
and `hands login --console` does the same with an Anthropic Console key. In both cases,
it then reports which account the brain is logged into and how it is logged in.
The brain's tools, permission rules and mode, and MCP servers come from that directory, as
for any Claude Code installation: `~/.hands/brain/settings.json`, and
`CLAUDE_CONFIG_DIR=~/.hands/brain claude mcp add -s user ...`. hands adds only its own MCP
server, which the brain may use without asking, and its hooks. It keeps the account's
claude.ai connectors out.
No one is at the brain's keyboard to answer a dialog. When the brain's settings require
permission for a tool, hands asks the user by voice, during the user turn that led to the
request: hands describes what the tool would do, and the tool runs only if the user says a
plain yes after the question has finished playing. Any other response refuses the
permission and is passed to the brain. A permission request that cannot be put to anyone,
or that no one answers in time, is refused. A `brain.permission` event records how each
request was resolved. All other dialogs are refused: the brain asks its questions in its
reply, and an MCP server that requests input produces a `brain.elicitation` event. Its
settings set `"defaultMode": "default"`, with the tools it may use without asking listed
in `permissions.allow`. `"syncClaudeAiSkills": false` and `"syncClaudeAiPlugins": false`
keep out the account's skills and plugins, which Claude Code reads from no other place.
Each `brain.turn` event lists the tools offered in the turn's latest request.
The brain works on a project's backlog itself, running `lit` from its shell in the
project's repository, as `lit quickstart` in that repository describes. `read_backlog`
reports where in the repository to run `lit`.
The brain searches the web and reads pages with the `firecrawl` command from its shell,
using that command's own login, as described by a skill in its setup:
`~/.hands/brain/skills/firecrawl/SKILL.md`. Its settings include
`"deny": ["WebSearch", "WebFetch"]` so that Firecrawl is its only web access. As a skill,
it adds one line to each request until it is used (about 110 bytes as of 2026-10-03);
Firecrawl's MCP server would add the descriptions of all its tools to every request.
The brain's prompt names its setup directory, so a skill you ask it by voice to install,
change, or remove is a folder in `~/.hands/brain/skills`, never in your `~/.claude`. The
brain can use a new skill from its next turn, without a restart.
In addition, hands gives the brain its own skills, shipped with the code they support (`src/hands/brain/plugin`):
`hands:prompt` defines how the brain writes a draft for a session from what you said. It is a small subset of `laws:prompt` made for that task.
`hands:chat` defines how it talks with you. It is a small subset of `laws:chat` made for spoken replies.
`hands:start` defines how it starts a new session when you request one, using hands' `start_session` tool.
`hands:close` defines how it closes the sessions you are finished with, by name or all completed sessions, using `close_session`.
The brain is interactive Claude Code running under hands' own fritter, never `claude -p`.
hands types each turn into its input, stops a turn with Escape, and types nothing else
into it. The brain's hooks report when a turn starts and when it ends.
Background requests from hands, such as summaries and the sentence that replaces an old
tool result, are sent to a second interactive Claude Code instance started for each
question, with `/btw` and the question as its opening prompt
(`src/hands/brain/asides.py`). As a result, a spoken turn never waits for one. Each is
recorded as a `brain.aside` wide event. The brain is the pipeline's LLM stage
(`src/hands/brain/stage.py`). The speech is generated from its requests on the wire,
never from its screen. Each turn's `voice.turn` wide event lists the exchanges its words
came from. A barge-in stops the brain, except while a tool whose effect must complete is
running. That tool finishes, and its readback, or the reason it failed, is spoken. After
that, and after `stay_silent`, hands answers the brain's next request itself, so the
model is not asked to continue. A turn that the brain ends in error is reported as a
model failure.
The run's audit log records which backend, model, and account it connected to. The gate is
push-to-talk: the key serves as the voice activity detector and the microphone mute, so
the key defines the turn boundary and the pipeline can never transcribe its own output.
Measured on 2026-09-12, voice to voice with a local Qwen3-30B-A3B (since retired): 1.4 s
from key release to first audio on a plain turn, and 4.3 s on a turn with a tool call.
Measured on 2026-10-04 with the brain, over the same six spoken turns: Whisper on MLX in
hands produced the transcript 298 to 329 ms after the release, with first audio at 1.3 to
1.75 s on a plain turn. LowTalker's Whisper, the same large-v3-turbo model on the Neural
Engine, took 613 to 724 ms, with first audio at 1.5 to 2.0 s, both by upload and through
its Realtime socket, so hands keeps its own Whisper.

## Installing the hooks

hands receives events from a session through Claude Code hooks, which come from a plugin.
This repository is a marketplace whose only entry is that plugin. Installing it installs
the hooks in every session, without manually merging anything into your settings. The
plugin itself comes from the hands you installed: the entry has Claude Code run
`hands plugin`, which writes the plugin, with hooks that run on that hands' own Python,
and prints its location. Claude Code shows you that command and asks you to accept it
during installation.

```
hands install-plugin                              # the single install command runs this; accept `hands plugin`; enables hooks in every new session
claude plugin disable hands@cc-hands              # disables hooks; the plugin stays installed
claude plugin enable hands@cc-hands               # re-enables hooks
claude plugin uninstall hands@cc-hands            # removes hooks
claude plugin marketplace remove cc-hands         # also removes the marketplace
```

The same commands are available as `/plugin ...` inside a session. Claude Code runs
`hands plugin` again once in every session, shortly after it starts. As a result, after
hands is upgraded, the next session uses the new version's hooks and skills without any
plugin command. A session that is already running picks up the change on
`/reload-plugins`. `hands` must be on the `PATH` that Claude Code runs with. For a
checkout, that must be a `hands` that runs `uv run --project <checkout> hands`. A
checkout's hooks run the checkout's own `src`, through its venv.

The plugin also contains two skills. `/hands:restart` restarts a running hands from your
current session, so that it runs the code, the brain's prompt, and the brain's setup
currently on disk. hands stops as it does when you press `q`, and starts again in the
same terminal and process, with the same menu bar item and all running sessions still
listed. The skill confirms this in one line when the new run is up, usually within
seconds. No change takes effect without a restart. A skill added to the plugin reaches a
session on `/reload-plugins`.

`/hands:attention` sets what hands announces without being asked. You can also ask hands
for this in your own words ("stop telling me when sessions finish", "be quiet for a
while"). It controls announcements for each finished turn (`finished off|brief|full`), the
focused session's steps as it works (`progress off|brief|full`), and a session ending
(`ended on|off`). Each of Claude Code's other hooks also has a setting, `off|brief|full`,
named after the hook: `permission_denied` (auto mode refused a call), `subagent_start`,
`subagent_stop`, `task_completed`, `config_change`, `pre_compact`, and `clear` (a
`/clear`, never a session's first start). Brief reports what happened. Full also includes
the hook's details, such as the subagent's report or the reason auto mode gave.
`quiet on` holds all announcements until `quiet off`, and leaves the other settings
unchanged. With no arguments, it reports the current settings. Everything except `ended`
is off until you turn it on. While finished turns are off, hands announces finished turns
only for sessions you asked it to watch, and reports any session's last turn when you ask
for it. Permission requests, questions asked in a dialog, and plans are always spoken,
regardless of these settings. Settings persist across restarts and take effect from the
next announcement hands would have made.

hands speaks with Charles, one of Pocket TTS's built-in voices, until you choose another.
Ask it which voices are available and to play some of them: each one says a line in its
own voice. Tell it which one to use, and it uses that voice from its next response. The
choice is stored in `~/.hands/voice`, so it persists across restarts.

The shim locates hands' home directory the same way the CLI does: `HANDS_HOME`, which must
be an absolute path, or `~/.hands`. `hands plugin` writes the plugin there, in
`plugins/`, with one directory for each interpreter and each change to the plugin's files.

## Wrapping every session

hands types into a session through fritter. For a session started outside fritter, it
types through tmux into the pane the session runs in. hands cannot type into a session
that runs outside both fritter and tmux. Install a `claude` that starts every session
under fritter, and put it first on `PATH`:

```
uv run hands install-fritter                    # copies fritter and writes ~/.hands/bin/claude next to it
export PATH="$HOME/.hands/bin:$PATH"            # add to your shell's startup file
```

That `claude` runs the next `claude` on `PATH` under fritter when both standard input and
standard output are a terminal, its first argument is not a subcommand, and there is no
`-p` or `--print`. A pipe, a script, `claude update`, and `claude -p` run the real claude
unchanged. A first argument that is a single bare lowercase word is treated as a
subcommand, so an opening prompt of one such word runs outside fritter: hands cannot type
into `claude review`, but it can type into `claude "review this"`.
`hands install-fritter` exits with code 0 only if the `claude` on the current `PATH` is
the one it wrote; otherwise, it reports what to add. The fritter it copies is the one
included in hands' package, built when that package was built, so Go is not required to
run it. Run it again after upgrading hands, or, in a checkout, after fritter's source
changes: `hands run` refuses to start if the copy does not match the one this hands
includes, and `hands check` reports when the copy, or the shim, does not match what this
hands would install. A session started before the shim was installed stays unreachable
until it ends.

A session under fritter connects to the API through its own fritter, which acts as the
session's proxy. The session still uses the API address it would have used. fritter
answers the session's connections to that API's host with its own certificate, which only
the session is configured to trust, forwards each request, and sends hands a copy of the
exchange. As a result, the audit log contains every request each session makes, under its
session id. All other connections pass through fritter without being inspected. Because
the session still uses Anthropic's API address, Claude Code keeps every feature it
provides for that API, including Remote Control. The session never waits for hands: while
hands is stopped, its requests go through as before, and the number of copies it could not
deliver is included in the next copy hands receives. The session's shell commands, and
any `claude` started from it, use the proxy the session had before fritter. A firewall
that prompts per program, such as Little Snitch, sees the session's connections as
fritter's and holds each one until you respond: allow outbound connections for
`~/.hands/bin/fritter`, and allow them again after `install-fritter` copies a new one,
which happens only when hands itself is a new build.

The hooks are active whether or not hands is running. While hands is stopped, has never
run, or is still starting, the hooks have no effect on a session: there is no hook error,
and a permission request shows Claude Code's own dialog. If hands crashed, hung, or left a
heartbeat that cannot be read, every session shows a hook error that reports it.

`src/hands/sessions/plugin/hooks/hooks.json` is generated from `hands.sessions.hookconfig`, and a
test fails if the two differ. After changing the hook table, run:

```
uv run python -m hands.sessions.hookconfig > src/hands/sessions/plugin/hooks/hooks.json
```

## Status and logs

`hands run` starts the menu bar indicator alongside it. The indicator's title is ✋ while
the daemon is up. It shows "hands stuck", "hands down", "hands refused to start", or "hands unreadable"
when something is wrong, and "✋ off" when the daemon was stopped or never ran. It posts a
notification when the daemon stops being up. When the daemon is gone, it posts a
notification and exits.

In tmux, the same title appears in the status line: green while up, yellow when stuck or
degraded, red when down, refused, or unreadable, and grey when off. Add these lines to
`~/.tmux.conf`. They set tmux's own `status-right` with the hands segment first, because
tmux truncates the end at `status-right-length`, and show a red warning in its place when
tmux cannot run `hands` (for example, when it is not on the tmux server's PATH). tmux
redraws it every `status-interval` (15 s unless set):

```
set -g status-right-length 80
set -g status-right '#(hands tmux-status || echo "#[fg=red,bold]⚠︎ hands tmux-status failed#[default]") "#{=21:pane_title}" %H:%M %d-%b-%y'
```

Every heartbeat rewrites `~/.hands/status.json`. Every effect and failure is written as a
line in the segmented log `~/.hands/audit/` (at most two 32 MiB segments), and the daemon's output goes to its terminal. Each line's
`level` is `error` for anything that went wrong, so `jq -cR 'fromjson? | select(.level == "error")'`
finds all of them. The brain is given the same information and reads the log itself when
asked what happened.

## The packaged app

hands.app starts hands only with a license key that Polar reports as active. On first
start, it prompts for the key from the user's purchase, stores it in ~/Library/Application
Support/hands/license.json, and verifies it with Polar on every start. If Polar rejects
the key, because it was revoked when a subscription ended or because it expired, hands
does not start, and the app displays Polar's reason. When Polar cannot be reached, the app
starts hands for 14 days after Polar last reported the key as active. The check sends
Polar only the key and the organization's id.

Before it starts hands, the app checks that macOS has given it the Microphone and Input
Monitoring permissions. If either is missing, it opens a setup window with one step per
permission. Each step says what the permission lets hands do, and its Next button shows
macOS's own request for it. For Input Monitoring, that request offers to open System
Settings, where hands is already listed and the user turns it on. A step moves on once macOS
reports the permission as granted. If the user declines or closes the request, Next resets
hands' entry with `tccutil` so that macOS shows the request again. `tccutil` finds the app
through Spotlight, so this works only for a copy of hands.app in a folder Spotlight indexes,
such as /Applications. When both permissions are granted, the window closes and hands starts.
Closing the window quits the app. The app asks macOS about its permissions from a new
process each time (`hands.app/Contents/MacOS/hands --permissions`), because a running process
can keep the answer it had before Input Monitoring was turned on.

## Tests and evals

`pytest` and `pyright` check the code. The eval checks what a listener hears: it runs four
real turns, taken unchanged from real transcripts, and checks that the facts are present,
that no code identifiers are spoken, that the spoken output stays within its configured
length, and that every number the model said appears in the turn. It exits with code 0
when every check passes, 1 when a check fails, and 2 when the model could not be reached
at all.

The intermediary's eval checks its next action at a single point in a conversation: the
tool it calls and with which arguments, or what it says (checked by the same spoken-output
judge), or that it stays silent for speech not directed at it. Its exit codes have the
same meanings.

## Shape

hands is one Python process, `hands`, run in a terminal. It has a Pipecat voice pipeline
on one side and the Claude Code integration on the other, connected through a pure core.

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

## Where the design lives

- [docs/needs.md](needs.md): the nine requirements that must be met before a developer
  can leave the keyboard during a working session.
- [docs/architecture.md](architecture.md): the daemon, its four packages, the
  types at every interface, the three speech channels, how hooks and transcripts are used,
  and how failures are kept visible.
- [docs/features.md](features.md): the epics that deliver the needs, each with its
  definition of done; `lit backlog` holds the build order.
- [fritter/README.md](../fritter/README.md): the pty wrapper that types into a session,
  its protocol, and the measurements taken against Claude Code.
- [docs/failure-modes.md](failure-modes.md): observed and anticipated failures, and the
  rule derived from each entry.
- [docs/happy-voice-reference.md](happy-voice-reference.md): how Happy, the
  closest prior art, works, and what to copy and avoid.
- [design-docs/monetization/hosted-subscription.md](../design-docs/monetization/hosted-subscription.md):
  what a paid version of hands can sell under Anthropic's terms, and what has to change first.

## Prior art

Happy (`~/code/happy`) provides hands-free Claude Code over ElevenLabs ConvAI. Its core
workflow works, which shows that this approach is worth building. Its problems were with
transport reliability, not content. Its two ideas worth keeping are the split between the
silent-context channel and the speaks-now channel, and the read watermark. The full
analysis is in [docs/happy-voice-reference.md](happy-voice-reference.md).
