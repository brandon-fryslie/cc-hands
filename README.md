# cc-hands

Talk to Claude Code instead of typing. Hold Right Shift and say what you want. hands writes
the prompt, types it into the correct session, and reads the result aloud. You can also ask
for more detail, answer permission prompts, and switch sessions by voice.

## Install

hands runs on macOS on Apple silicon. Run this in Terminal:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/promptctl/cc-hands/master/install.sh)"
```

The installer runs nine numbered steps, skips any step that is already complete, and then
starts hands in the same terminal. It stops only for the following steps, which you must
complete yourself. Each one is explained before it is requested:

- Your administrator password, if Homebrew isn't installed yet.
- Claude Code's first-run questions (theme, sign-in, folder trust), if you have not used
  Claude Code before, and whether to use your `ANTHROPIC_API_KEY` if one is set. Answer
  them, then type `/exit`.
- A browser sign-in for the brain, the separate copy of Claude Code that hands uses, with
  your Claude plan.
- Typing `y` when Claude Code asks whether to run `hands plugin`.
- The Input Monitoring permission for your terminal app, in System Settings, so hands can
  detect Right Shift in every app. If macOS offers to quit the app, choose Later.

If you close one of these before it finishes, the installer offers to retry it
immediately. Tool output is written to `~/Library/Logs/hands-install.log`. If a step fails,
the installer shows the last lines of that log.

## Use

Hold Right Shift in any app, speak, and release the key. hands responds aloud. The first
start downloads Whisper's speech model (about 1.6 GB). To quit hands, press `q` in its
terminal. To start it again, run `hands run`.

Start Claude Code sessions as usual by running `claude` in a new terminal. The installer
configures `claude` so that hands can type into these sessions. You can also ask hands to
start a session.

If something isn't working, run `hands check`. It checks each part of the setup and shows
the command that fixes any part that is missing.

## More

[docs/guide.md](docs/guide.md) covers settings, all commands, the brain, the plugin, and
how hands connects to your sessions. The design is described in
[docs/architecture.md](docs/architecture.md), and how it is tested in
[docs/testing.md](docs/testing.md).
