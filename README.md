# cc-hands

Talk to Claude Code instead of typing to it. Hold Right Shift, say what you want, and hands
writes the prompt, types it into the right session, and tells you what came back. Ask for
more detail, answer a permission prompt, or switch sessions, all by voice.

## Install

hands runs on macOS on Apple silicon. Run this in Terminal:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/promptctl/cc-hands/master/install.sh)"
```

It installs whatever is missing and ends with hands running in that terminal. Along the
way it stops for the few things only you can do, and says what each one is before it asks:

- Your administrator password, if Homebrew isn't installed yet.
- Claude Code's first-run questions (theme, login, folder trust), if your own Claude Code
  hasn't run before, along with whether to use an `ANTHROPIC_API_KEY` you have set. Answer
  them, then type `/exit`.
- A sign-in in your browser for the brain, the separate Claude Code that hands talks
  through, on your Claude plan.
- `y` when Claude Code asks whether to run `hands plugin`.
- The Input Monitoring grant for your terminal app, in System Settings, so hands hears
  Right Shift in every app. If macOS offers to quit the app, choose Later.

If the command stops part-way, run it again; it picks up where it left off.

## Use

Hold Right Shift in any app, talk, and let go. hands answers out loud. The first start
downloads Whisper's speech model, about 1.6 GB. Press `q` in hands' terminal to quit it,
and run `hands run` to start it again.

Start Claude Code sessions as usual with `claude` in a new terminal; the installer made
`claude` start sessions that hands can type into. You can also ask hands to start one.

If something isn't working, `hands check` lists each piece of the setup and what fixes the
one that's missing.

## More

[docs/guide.md](docs/guide.md) covers settings, every command, the brain, the plugin, and
how hands reaches your sessions. The design is in [docs/architecture.md](docs/architecture.md).
