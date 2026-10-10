#!/bin/bash
# install.sh: take a Mac to an installed hands, in one command the README gives:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/promptctl/cc-hands/master/install.sh)"
#
# A guided run: it opens on what it will do and what it will ask, then walks nine numbered steps, each one ticked off
# when done, and ends running hands in this terminal, then in a fresh login shell once hands quits, so this terminal has
# the new PATH too. What it installs is installed only when missing: Claude Code, Homebrew, PortAudio, uv, the newest
# released hands, the claude shim that starts every session under fritter, and hands' plugin. The tools' own output goes
# to a log, shown when one of them fails. The only input it asks for is the administrator password Homebrew's own install
# needs, what Claude Code asks only once (its theme, its login, and whether to trust the folder `hands smoke` runs in),
# the same for the brain, hands' own Claude Code, the yes Claude Code asks for to install a plugin by running a command,
# and the Input Monitoring grant macOS keeps for the person to give in System Settings. Each is explained before it is
# asked, and one the person leaves unfinished is offered again on the spot; a run stopped part-way is finished by running
# it again.
set -euo pipefail

REPO=promptctl/cc-hands
# [LAW:one-source-of-truth] hatch_build.TAG, the platform tag every release's wheel carries; tests hold the two equal.
WHEEL_TAG=py3-none-macosx_12_0_arm64
# Homebrew's own name for where it lives, which is /opt/homebrew on Apple silicon.
HOMEBREW_PREFIX=${HOMEBREW_PREFIX:-/opt/homebrew}
STEPS=9

# Whether a person is at this terminal to be asked: the README's command runs with the terminal as its input.
interactive=false
[ -t 0 ] && [ -t 1 ] && interactive=true
if [ -t 1 ]; then
  bold=$'\033[1m' dim=$'\033[2m' green=$'\033[32m' red=$'\033[31m' reset=$'\033[0m'
else
  bold='' dim='' green='' red='' reset=''
fi

step=0
title() { step=$((step + 1)); printf '\n%s[%d/%d] %s%s\n' "$bold" "$step" "$STEPS" "$*" "$reset"; }
note() { printf '      %s\n' "$*"; }
# A path as the person would type it.
tilde() { printf '%s' "${1/#$HOME/~}"; }
ok() { printf '      %s✓%s %s\n' "$green" "$reset" "$*"; }
fail() { printf '\n      %s✗ %s%s\n' "$red" "$*" "$reset" >&2; exit 1; }
# Waits for the person before what they must do next fills the screen: the explanation above stays where they read it.
pause() { if $interactive; then printf '      %sPress Enter to continue.%s ' "$dim" "$reset"; read -r _; fi; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "hands runs on macOS on Apple silicon; this is $(uname -s) $(uname -m)"

# Every tool's own output lands here, so the steps read as steps; a tool that fails has its last lines shown from it.
LOG_FILE=$HOME/Library/Logs/hands-install.log
mkdir -p "${LOG_FILE%/*}"
printf '\n==== hands install, %s\n' "$(date)" >>"$LOG_FILE"

# Runs a tool with its output in the log, a spinner and the time so far in its place; on failure, says what failed with
# the log's last lines and where the rest is.
quietly() {
  local what=$1 code=0 started=$SECONDS turn=0 frames='-\|/'
  shift
  printf '\n$ %s\n' "$*" >>"$LOG_FILE"
  "$@" >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  if [ -t 1 ]; then
    # kill -0 asks only whether the tool still runs; what it prints when the tool has exited is that answer, not an error.
    while kill -0 "$pid" 2>/dev/null; do
      printf '\r      %s %s... %ds ' "${frames:turn++%4:1}" "$what" $((SECONDS - started))
      sleep 0.2
    done
    printf '\r\033[K'
  fi
  wait "$pid" || code=$?
  [ "$code" -eq 0 ] && return
  printf '      %s\n' "$what failed (exit $code). The last of its output:" >&2
  tail -n 15 "$LOG_FILE" | sed 's/^/        /' >&2
  fail "$what failed. Its full output is in $(tilde "$LOG_FILE")"
}

# The login shell every new terminal starts, as the account records it: $SHELL is only the shell this ran from, which a
# person's zsh terminal and a tool's bash, say, do not share.
login_shell=$(dscl . -read "/Users/$(id -un)" UserShell) || fail "your login shell could not be read from your account (dscl failed)"
login_shell=${login_shell#UserShell: }
[ -x "$login_shell" ] || fail "your account's login shell, $login_shell, is not a program on this Mac"

printf '\n%shands installer%s\n\n' "$bold" "$reset"
note "This sets up hands, which lets you talk to Claude Code, on this Mac."
note "Most of it runs on its own. What only you can do, it explains first:"
note "  - your Mac password, if Homebrew isn't installed yet"
note "  - signing in to Claude in your browser: once for Claude Code, once for hands"
note "  - one \"y\" so Claude Code loads hands' plugin"
note "  - one switch in System Settings, so hands hears the Right Shift key"
note "Anything already set up is skipped. Tool output goes to $(tilde "$LOG_FILE")"
pause

# What this run installs is found here, whatever PATH the terminal it runs in has.
export PATH="$HOME/.local/bin:$HOMEBREW_PREFIX/bin:$PATH"

title "Claude Code"
# Claude Code's installer puts its native claude in ~/.local/bin, as `hands check` wants it.
if [ -x "$HOME/.local/bin/claude" ]; then
  ok "already installed"
else
  quietly "Installing Claude Code" /bin/bash -o pipefail -c 'curl -fsSL https://claude.ai/install.sh | bash'
  ok "installed"
fi

title "Homebrew"
if [ -x "$HOMEBREW_PREFIX/bin/brew" ]; then
  ok "already installed"
else
  note "Homebrew installs the audio library hands needs. Creating $HOMEBREW_PREFIX needs your"
  note "administrator password, the one you log in to this Mac with."
  sudo -v
  # Homebrew asks sudo without a prompt in its non-interactive mode, so the password given above is kept fresh until it
  # is done; its own install of the Command Line Tools can outlast sudo's five minutes. The keeper ends with this run
  # however it ends, a failed or interrupted install included, and is stopped here when Homebrew is in.
  while kill -0 $$ 2>/dev/null && sudo -n -v 2>/dev/null; do sleep 60; done &
  keeper=$!
  homebrew_installer=$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh) || fail "Homebrew's installer could not be downloaded: check this Mac's internet connection"
  note "This can take 5 to 10 minutes, mostly Apple's Command Line Tools."
  quietly "Installing Homebrew" env NONINTERACTIVE=1 /bin/bash -c "$homebrew_installer"
  kill "$keeper" 2>/dev/null || true
  ok "installed"
fi

title "PortAudio and uv"
# [LAW:dataflow-not-control-flow] each formula is installed only when missing: `brew install` of an outdated one upgrades it,
# and a second run changes no version.
for formula in portaudio uv; do
  if brew list --formula "$formula" >/dev/null 2>&1; then
    ok "$formula already installed"
  else
    quietly "Installing $formula" brew install "$formula"
    ok "$formula installed"
  fi
done

# The newest release is the tag github.com/<repo>/releases/latest redirects to.
latest=$(curl -fsSI -o /dev/null -w '%{redirect_url}' "https://github.com/$REPO/releases/latest") || fail "hands' newest release could not be looked up on GitHub: check this Mac's internet connection"
tag=${latest##*/}
# [LAW:parse-dont-validate] a release tag is vX.Y.Z exactly, as release.yml publishes it.
echo "$tag" | grep -Eqx 'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)' || fail "the newest release of $REPO is not a vX.Y.Z tag: $latest"
version=${tag#v}
title "hands $version"
bin=$(uv tool dir --bin)
if [ -x "$bin/hands" ] && [ "$("$bin/hands" --version 2>/dev/null)" = "hands $version" ]; then
  ok "already installed, the newest release"
else
  release=https://github.com/$REPO/releases/download/$tag
  # A Mac has its own Python, 3.9, which uv would otherwise take; hands needs 3.12, which uv fetches. The constraints are
  # the versions the release was tested on. --reinstall replaces a hands that is there but does not answer its version.
  quietly "Installing hands $version" uv tool install --reinstall --python 3.12 --constraints "$release/constraints.txt" "$release/hands-$version-$WHEEL_TAG.whl"
  ok "installed"
fi
# Claude Code runs `hands plugin` from this PATH to install the plugin.
export PATH="$bin:$PATH"

title "Your terminal"
# hands' home as hands finds it (hands.sessions.home.default_home): HANDS_HOME, or ~/.hands. Its bin holds the claude
# shim, which hands writes and which says it is installed only when it is the claude PATH finds.
shims=${HANDS_HOME:-$HOME/.hands}/bin
PATH="$shims:$PATH" quietly "Installing the claude shim" hands install-fritter

# The login shell's PATH is the one every new terminal starts with: a directory it lacks gets one line in the profile
# it reads, and one it has is left as it is. A new terminal's shell is interactive too, and zsh then reads ~/.zshrc after
# the profile: rc is the file such a shell reads last.
case $(basename "$login_shell") in
  zsh) profile=$HOME/.zprofile rc=$HOME/.zshrc ;;
  bash)
    # A bash login shell reads only the first of these that exists, so the line goes in that one; with none, a new
    # .bash_profile.
    profile=$HOME/.bash_profile
    for name in .profile .bash_login .bash_profile; do [ ! -e "$HOME/$name" ] || profile=$HOME/$name; done
    rc=$profile ;;
  *) profile='' rc='' ;;
esac
# The PATH the person's shell, started with these flags, ends its startup files with. It starts from launchd's PATH, as a
# new terminal's does, not this run's, which already has every directory, and its PATH is the line marked as it, among
# whatever its startup files and a login shell's logout file print. It fails where the shell ends without saying it.
shell_path() {
  local printed
  printed=$(env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin "$login_shell" "$1" 'printf "\n@hands-path@%s\n" "$PATH"' </dev/null) || return
  printed=$(printf '%s\n' "$printed" | sed -n 's/^@hands-path@//p' | tail -n 1)
  [ -n "$printed" ] && printf '%s' "$printed"
}
login_path=$(shell_path -lc) || fail "your shell, $login_shell -lc, run with no terminal, ended without saying its PATH, so which commands a new terminal finds is unknown"
on_login_path() { case ":$login_path:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }
lines=()
on_login_path "$HOMEBREW_PREFIX/bin" || lines+=("eval \"\$($HOMEBREW_PREFIX/bin/brew shellenv)\"")
on_login_path "$HOME/.local/bin" || lines+=("export PATH=\"\$HOME/.local/bin:\$PATH\"")
[ "$bin" = "$HOME/.local/bin" ] || on_login_path "$bin" || lines+=("export PATH=\"$bin:\$PATH\"")
if [ ${#lines[@]} -gt 0 ]; then
  [ -n "$profile" ] || fail "your shell, $login_shell, is not zsh or bash; add these lines to its login profile and open a new terminal: ${lines[*]}"
  printf '%s\n' "${lines[@]}" >>"$profile"
  printf 'added to %s: %s\n' "$profile" "${lines[*]}" >>"$LOG_FILE"
  ok "updated $(tilde "$profile") so new terminals find Homebrew, Claude Code, and hands"
fi
# The shim runs the claude after it on PATH, so it must be the claude a new terminal finds, not only on its PATH: the
# shim's line goes last in the file read last. A shell whose startup files end it when it has no terminal, as one that
# starts tmux does, cannot be asked what a new terminal finds: the login shell is asked instead, and the line goes in the
# profile it reads.
terminal=-ilc
if ! shell_path -ilc >/dev/null; then
  terminal=-lc rc=$profile
  note "your shell's startup files end it when it has no terminal, as ones that start tmux do: which claude a new terminal finds is judged from its login shell alone, which reads $profile"
fi
terminal_claude() { PATH=$(shell_path "$terminal") command -v claude; }
line="export PATH=\"$shims:\$PATH\""
if [ "$(terminal_claude)" != "$shims/claude" ]; then
  [ -n "$rc" ] || fail "your shell, $login_shell, is not zsh or bash; add this line last to the file it reads at start and open a new terminal: $line"
  # A line there already has been read and undone; adding it again would undo nothing more.
  [ -f "$rc" ] && grep -qxF "$line" "$rc" || {
    printf '%s\n' "$line" >>"$rc"
    printf 'added to %s: %s\n' "$rc" "$line" >>"$LOG_FILE"
  }
  [ "$(terminal_claude)" = "$shims/claude" ] || fail "a new terminal's claude is $(terminal_claude), not hands' $shims/claude, though $rc has $line: a startup file read after it puts another claude first; put that line after this one"
fi
ok "\`claude\` in a new terminal starts sessions hands can type into"

# A step only the person can finish, as `hands <command>` asks it at this terminal: 1 is asked and not given, which is
# offered again here while a person is at it, and otherwise left to a second run of this command; 2 is that it could not
# be asked here, as the command says; any other exit is the command's own failure, as it says. Each is said as what is
# not done ($2), or why it cannot be done here ($3).
person() {
  local code
  while :; do
    code=0
    hands "$1" || code=$?
    case $code in
      0) return ;;
      1)
        $interactive || fail "$2. Run the install command again to pick up here."
        printf '\n      %sThat did not finish: %s.%s\n' "$red" "$2" "$reset"
        printf '      Press Enter to try it again, or Control-C to stop here. '
        read -r _ ;;
      2) fail "$3, as said above" ;;
      *) fail "$2: hands $1 failed, exiting $code, as said above" ;;
    esac
  done
}
# Whether `hands <command>` has nothing to ask: with no terminal to ask at, it exits 0 only when that is so.
answered() { hands "$1" </dev/null >>"$LOG_FILE" 2>&1; }

title "Sign in to Claude Code"
# Claude Code's first-run questions and its login, asked once here, so the session `hands smoke` starts waits on none of them.
if answered first-run; then
  ok "already signed in and set up"
else
  note "Claude Code opens in this window and asks its one-time questions: a color theme,"
  note "signing in to your Claude account in your browser, and whether to trust hands'"
  note "test folder. Answer them, then type /exit to come back here."
  pause
  person first-run "Claude Code's first run is not finished, so the sessions hands starts would wait on its questions" \
    "Claude Code's first run is not finished: Claude Code could not be asked"
  ok "signed in and set up"
fi

title "Sign in hands' brain"
# The brain's login, which only the person can make: a brain that holds one keeps it, and nothing is asked.
if answered login; then
  ok "already signed in"
else
  note "hands thinks with its own Claude Code, the brain, which signs in to your Claude plan"
  note "on its own. It opens in this window: answer what it asks and sign in in your"
  note "browser; type /exit if it leaves you at its prompt."
  pause
  person login "the brain is not signed in, so hands has no model to talk with" \
    "the brain is not signed in, so hands has no model to talk with: Claude Code could not be asked"
  ok "signed in"
fi

title "hands' Claude Code plugin"
# After the steps Claude Code asks for, so that declining it leaves every step before it done.
note "The plugin lets every Claude Code session tell hands what it is doing."
person install-plugin "hands' Claude Code plugin is not installed, so no session joins hands" \
  "hands' Claude Code plugin is not installed, so no session joins hands: Claude Code could not be asked"
ok "installed"

title "The talk key"
# The talk key's Input Monitoring grant, which only the person can give, to the app this runs in: hands waits for it.
person grant "hands cannot hear the talk key without the Input Monitoring grant" \
  "hands cannot hear the talk key: the Input Monitoring grant cannot be given here"
ok "hands hears Right Shift in every app"

printf '\n%s%sAll set.%s hands %s is installed.\n\n' "$bold" "$green" "$reset" "$version"
# [LAW:one-source-of-truth] hands starts its sessions from the PATH a new terminal has, as `hands check` there judges it,
# and it is the hands this run installed, whatever that PATH finds first.
run_path=$(shell_path "$terminal") || fail "your shell, $login_shell $terminal, run with no terminal, ended without saying its PATH, so hands cannot start its sessions from a new terminal's"
note "Starting hands. Hold Right Shift in any app, say what you want, and let go."
note "The first start downloads its speech model, about 1.6 GB. Press q here to quit;"
note "\`hands run\` starts it again. Start Claude Code with \`claude\` in a new terminal."
printf '\n'
ran=0
PATH=$run_path "$bin/hands" run || ran=$?
case $ran in
  0) ;;
  # [LAW:single-enforcer] whether a hands runs already is the home's lock's to say: `hands run` exits 3 for it.
  3) note "hands is running already, so this run starts no second one; this terminal is now a login shell that finds it" ;;
  *) fail "hands stopped, exiting $ran, as said above; \`hands run\` starts it again" ;;
esac
exec "$login_shell" -l
