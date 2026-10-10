#!/bin/bash
# install.sh: installs hands on a Mac. The README gives the command that runs it:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/promptctl/cc-hands/master/install.sh)"
#
# The installer first lists what it will ask the user for. It then runs nine numbered steps, marks each one done, and
# skips any step that is already complete. It installs Claude Code, Homebrew, PortAudio, uv, the latest hands release,
# the claude shim (which starts every session under fritter), and hands' plugin, and then starts hands in this terminal.
# When hands quits, the installer replaces itself with a new login shell so this terminal has the updated PATH.
#
# Tool output goes to a log file. If a tool fails, the end of the log is shown. The installer stops for input only where
# the user must act: the administrator password (for Homebrew), Claude Code's first-run questions and sign-in, the
# brain's sign-in, the confirmation Claude Code requires to install the plugin, and the Input Monitoring permission in
# System Settings. Each one is explained before it is requested. If the user does not finish one, the installer offers
# to retry it immediately. If the installer is stopped, running it again continues where it left off.
set -euo pipefail

REPO=promptctl/cc-hands
# [LAW:one-source-of-truth] Must match hatch_build.TAG, the platform tag on every release wheel. A test checks this.
WHEEL_TAG=py3-none-macosx_12_0_arm64
# Homebrew's install location. On Apple silicon this is /opt/homebrew.
HOMEBREW_PREFIX=${HOMEBREW_PREFIX:-/opt/homebrew}
STEPS=9

# The installer can ask questions only when both input and output are a terminal, as with the README's command.
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
# Shortens a path under the home directory to ~/... for display. The ~ comes from a variable because bash 4.3 and later
# expand a literal ~ in the replacement, and bash 3.2 prints an escaped one with its backslash.
tilde() { local home='~'; printf '%s' "${1/#$HOME/$home}"; }
ok() { printf '      %s✓%s %s\n' "$green" "$reset" "$*"; }
fail() { printf '\n      %s✗ %s%s\n' "$red" "$*" "$reset" >&2; exit 1; }
# Discards keys typed earlier (for example, Enter pressed during a long install), so they do not answer the next prompt.
# Python is not available on a new Mac, but perl is.
flush_input() { perl -MPOSIX -e 'tcflush(0, TCIFLUSH)' 2>/dev/null || true; }
# Prints a prompt and waits for Enter. End of input (Control-D) stops the installer.
wait_for_enter() {
  flush_input
  printf '      %s%s%s ' "$dim" "$1" "$reset"
  read -r _ || fail "Stopped. Run the install command again to continue from here."
}
# Waits before a step takes over the screen, so the user can read the explanation first.
pause() { if $interactive; then wait_for_enter "Press Enter to continue."; fi; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "hands requires macOS on Apple silicon. This Mac is $(uname -s) $(uname -m)."

# All tool output is written here. When a tool fails, the last lines of this file are shown.
LOG_FILE=$HOME/Library/Logs/hands-install.log
mkdir -p "${LOG_FILE%/*}"
printf '\n==== hands install, %s\n' "$(date)" >>"$LOG_FILE"

log_tail() {
  tail -n 15 "$LOG_FILE" | sed 's/^/        /' >&2
}

# Runs a tool with its output sent to the log, and shows a spinner with the elapsed time. If the tool fails, shows the
# end of the log and stops.
quietly() {
  local what=$1 code=0 started=$SECONDS turn=0 frames='-\|/'
  shift
  printf '\n$ %s\n' "$*" >>"$LOG_FILE"
  # Job control (set -m) starts the tool in its own process group, so that Control-C can stop the tool and every
  # process it started. Without it, a background job ignores Control-C and keeps running after the installer exits.
  set -m
  "$@" >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  set +m
  trap 'kill -TERM -- "-$pid" 2>/dev/null; trap - INT TERM; fail "$what was stopped. Run the install command again to continue from here."' INT TERM
  if [ -t 1 ]; then
    # kill -0 only checks whether the tool is still running. Its error output after the tool exits is discarded.
    while kill -0 "$pid" 2>/dev/null; do
      printf '\r      %s %s... %ds ' "${frames:turn++%4:1}" "$what" $((SECONDS - started))
      sleep 0.2
    done
    printf '\r\033[K'
  fi
  wait "$pid" || code=$?
  trap - INT TERM
  [ "$code" -eq 0 ] && return
  printf '      %s\n' "$what failed (exit code $code). The last lines of its output:" >&2
  log_tail
  fail "$what failed. The full output is in $(tilde "$LOG_FILE")"
}

# The login shell that new terminals start, as recorded for the user's account. $SHELL is only the shell this script
# was started from, which can differ (for example, a tool running bash for a user whose terminals run zsh). The search
# node also covers network accounts. A value containing a space is printed on the line after the key, so both lines
# are read. If the lookup fails, $SHELL is used.
login_shell=$(dscl /Search -read "/Users/$(id -un)" UserShell 2>/dev/null | sed -e 's/^UserShell:[[:space:]]*//' -e 's/^[[:space:]]*//' | tr -d '\n') || login_shell=''
[ -n "$login_shell" ] || login_shell=${SHELL:-}
[ -x "$login_shell" ] || fail "Your account's login shell, '$login_shell', is not a program on this Mac."

printf '\n%shands installer%s\n\n' "$bold" "$reset"
note "This sets up hands, which lets you talk to Claude Code, on this Mac."
note "Most of it runs on its own. It explains each thing only you can do before asking:"
note "  - your Mac password, if Homebrew isn't installed yet"
note "  - signing in to Claude in your browser: once for Claude Code, once for hands"
note "  - one \"y\" so Claude Code loads hands' plugin"
note "  - one switch in System Settings, so hands hears the Right Shift key"
note "Anything already set up is skipped. Tool output goes to $(tilde "$LOG_FILE")"
pause

# Adds the install locations to PATH for this run, regardless of the terminal's PATH.
export PATH="$HOME/.local/bin:$HOMEBREW_PREFIX/bin:$PATH"

title "Claude Code"
# Claude Code's installer puts the native claude in ~/.local/bin, which is where `hands check` expects it.
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
  # In non-interactive mode, Homebrew runs sudo without a password prompt. This loop keeps the sudo credentials valid
  # until Homebrew finishes, because its Command Line Tools install can take longer than sudo's five-minute timeout.
  # The loop exits when this script exits for any reason, and is stopped below once Homebrew is installed.
  while kill -0 $$ 2>/dev/null && sudo -n -v 2>/dev/null; do sleep 60; done &
  keeper=$!
  homebrew_installer=$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh) || fail "Homebrew's installer could not be downloaded. Check this Mac's internet connection."
  note "This can take 5 to 10 minutes, mostly for Apple's Command Line Tools."
  quietly "Installing Homebrew" env NONINTERACTIVE=1 /bin/bash -c "$homebrew_installer"
  kill "$keeper" 2>/dev/null || true
  ok "installed"
fi

title "PortAudio and uv"
# [LAW:dataflow-not-control-flow] Each formula is installed only if it is missing. `brew install` would upgrade an
# outdated formula, and a second run should not change any versions.
for formula in portaudio uv; do
  if brew list --formula "$formula" >/dev/null 2>&1; then
    ok "$formula already installed"
  else
    quietly "Installing $formula" brew install "$formula"
    ok "$formula installed"
  fi
done

# The latest release is the tag that github.com/<repo>/releases/latest redirects to.
latest=$(curl -fsSI -o /dev/null -w '%{redirect_url}' "https://github.com/$REPO/releases/latest") || fail "The latest hands release could not be found on GitHub. Check this Mac's internet connection."
tag=${latest##*/}
# [LAW:parse-dont-validate] A release tag must be exactly vX.Y.Z, the format release.yml publishes.
echo "$tag" | grep -Eqx 'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)' || fail "The latest release of $REPO is not a vX.Y.Z tag: $latest"
version=${tag#v}
title "hands $version"
bin=$(uv tool dir --bin)
if [ -x "$bin/hands" ] && [ "$("$bin/hands" --version 2>/dev/null)" = "hands $version" ]; then
  ok "already installed, the latest release"
else
  release=https://github.com/$REPO/releases/download/$tag
  # macOS includes Python 3.9, which uv would otherwise use. hands needs Python 3.12, which uv downloads. The constraints
  # file pins the dependency versions the release was tested with. --reinstall replaces an existing hands that does not
  # report its version.
  quietly "Installing hands $version" uv tool install --reinstall --python 3.12 --constraints "$release/constraints.txt" "$release/hands-$version-$WHEEL_TAG.whl"
  ok "installed"
fi
# Claude Code runs `hands plugin` from this PATH when it installs the plugin.
export PATH="$bin:$PATH"

title "Your terminal"
# hands' home directory, located the same way as hands.sessions.home.default_home: $HANDS_HOME, or ~/.hands. Its bin
# directory holds the claude shim. `hands install-fritter` writes the shim, and reports success only if it is the claude
# found on PATH.
shims=${HANDS_HOME:-$HOME/.hands}/bin
PATH="$shims:$PATH" quietly "Installing the claude shim" hands install-fritter

# New terminals start with the login shell's PATH. Each directory missing from it gets one line in the profile the login
# shell reads; directories already present are left alone. New terminal shells are also interactive, so zsh reads
# ~/.zshrc after the profile. $rc is the file the shell reads last.
case $(basename "$login_shell") in
  zsh) profile=$HOME/.zprofile rc=$HOME/.zshrc ;;
  bash)
    # A bash login shell reads only the first of these files that exists, so the line goes in that file. If none
    # exists, a new .bash_profile is created.
    profile=$HOME/.bash_profile
    for name in .profile .bash_login .bash_profile; do [ ! -e "$HOME/$name" ] || profile=$HOME/$name; done
    rc=$profile ;;
  *) profile='' rc='' ;;
esac
# Prints the PATH that the user's shell, started with the given flags, has after running its startup files. The shell
# starts from launchd's default PATH, as a new terminal does, not from this run's PATH. The startup files may print other
# output, so the PATH is printed on a marked line and read from there. Fails if the shell exits before printing it.
shell_path() {
  local printed
  printed=$(env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin "$login_shell" "$1" 'printf "\n@hands-path@%s\n" "$PATH"' </dev/null) || return
  printed=$(printf '%s\n' "$printed" | sed -n 's/^@hands-path@//p' | tail -n 1)
  [ -n "$printed" ] && printf '%s' "$printed"
}
login_path=$(shell_path -lc) || fail "Your shell ($login_shell -lc) exited without printing its PATH, so the installer cannot tell what a new terminal will find."
on_login_path() { case ":$login_path:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }
lines=() found=()
on_login_path "$HOMEBREW_PREFIX/bin" || { lines+=("eval \"\$($HOMEBREW_PREFIX/bin/brew shellenv)\""); found+=(Homebrew); }
on_login_path "$HOME/.local/bin" || { lines+=("export PATH=\"\$HOME/.local/bin:\$PATH\""); found+=("Claude Code"); }
[ "$bin" = "$HOME/.local/bin" ] || on_login_path "$bin" || { lines+=("export PATH=\"$bin:\$PATH\""); found+=(hands); }
if [ ${#lines[@]} -gt 0 ]; then
  [ -n "$profile" ] || fail "Your shell, $login_shell, is not zsh or bash. Add these lines to its login profile, then open a new terminal: ${lines[*]}"
  printf '%s\n' "${lines[@]}" >>"$profile"
  printf 'added to %s: %s\n' "$profile" "${lines[*]}" >>"$LOG_FILE"
  case ${#found[@]} in
    1) names=${found[0]} ;;
    2) names="${found[0]} and ${found[1]}" ;;
    *) names="${found[0]}, ${found[1]}, and ${found[2]}" ;;
  esac
  ok "updated $(tilde "$profile") so new terminals find $names"
fi
# The shim runs the next claude on PATH, so the shim must be the first claude a new terminal finds, not just present on
# PATH. Its line therefore goes at the end of the file the shell reads last. Some startup files exit when there is no
# terminal (for example, ones that start tmux). In that case the installer checks the login shell instead, and puts the
# line in the profile the login shell reads.
terminal=-ilc
if ! shell_path -ilc >/dev/null; then
  terminal=-lc rc=$profile
  note "Your shell's startup files exit when there is no terminal (as files that start tmux do), so the installer checks your login shell instead, which reads $(tilde "$profile")."
fi
terminal_claude() { PATH=$(shell_path "$terminal") command -v claude; }
line="export PATH=\"$shims:\$PATH\""
if [ "$(terminal_claude)" != "$shims/claude" ]; then
  [ -n "$rc" ] || fail "Your shell, $login_shell, is not zsh or bash. Add this line at the end of the file it reads at startup, then open a new terminal: $line"
  # If the line is already in the file, it is not added again.
  if ! { [ -f "$rc" ] && grep -qxF "$line" "$rc"; }; then
    printf '%s\n' "$line" >>"$rc"
    printf 'added to %s: %s\n' "$rc" "$line" >>"$LOG_FILE"
    ok "updated $(tilde "$rc") so \`claude\` in a new terminal starts hands' shim"
  fi
  [ "$(terminal_claude)" = "$shims/claude" ] || fail "A new terminal finds $(terminal_claude) as claude instead of hands' $shims/claude, even though $(tilde "$rc") contains $line. A startup file read after it puts another claude first. Move that line after this one."
fi
ok "\`claude\` in a new terminal starts sessions hands can type into"

# Runs `hands <command>` at this terminal for a step only the user can finish. Exit codes from hands: 0 means done;
# 1 means the user did not finish (retried here if a user is present, otherwise the installer stops); 2 means it cannot be
# asked here; any other code is a failure. $2 describes what is not done; $3 says why it cannot be done here.
person() {
  local code
  while :; do
    code=0
    hands "$1" || code=$?
    case $code in
      0) return ;;
      1)
        $interactive || fail "$2. Run the install command again to continue from here."
        printf '\n      %sThat did not finish: %s.%s\n' "$red" "$2" "$reset"
        wait_for_enter "Press Enter to try it again, or Control-C to stop here." ;;
      2) fail "$3. See the message above." ;;
      *) fail "$2: hands $1 failed with exit code $code. See the message above." ;;
    esac
  done
}
# Runs `hands <command>` with no terminal, which only checks whether the step needs input. Returns 0 if the step is
# already done, and 1 if it needs the user at this terminal. Stops the installer, showing the end of the log, if the
# step needs input but no user is present (exit code 2), or if the check fails in any other way. $2 describes what
# is not done; $3 says why it cannot be done here.
done_already() {
  local code=0
  printf '\n$ hands %s </dev/null\n' "$1" >>"$LOG_FILE"
  hands "$1" </dev/null >>"$LOG_FILE" 2>&1 || code=$?
  case $code in
    0) return 0 ;;
    2) $interactive && return 1
       log_tail
       fail "$3." ;;
    *) log_tail
       fail "$2: hands $1 failed with exit code $code. The full output is in $(tilde "$LOG_FILE")" ;;
  esac
}

title "Sign in to Claude Code"
# Claude Code's first-run questions and its sign-in are completed here, so the session `hands smoke` starts does not
# stop on any of them.
first_run_missing="Claude Code's first run is not finished, so the sessions hands starts would stop on its questions"
if done_already first-run "$first_run_missing" "$first_run_missing: Claude Code could not be asked"; then
  ok "already signed in and set up"
else
  note "Claude Code opens in this window and asks its one-time questions: a color theme,"
  note "signing in to your Claude account in your browser, and whether to trust hands'"
  note "test folder. Answer them, then type /exit to come back here."
  pause
  person first-run "$first_run_missing" "$first_run_missing: Claude Code could not be asked"
  ok "signed in and set up"
fi

title "Sign in hands' brain"
# The brain's sign-in, which only the user can complete. If the brain is already signed in, nothing is asked.
brain_missing="the brain is not signed in, so hands has no model to talk with"
if done_already login "$brain_missing" "$brain_missing: Claude Code could not be asked"; then
  ok "already signed in"
else
  note "hands thinks with its own copy of Claude Code, called the brain, which signs in to"
  note "your Claude plan separately. Your browser opens: sign in there, then come back here."
  note "If Claude Code opens in this window instead, answer its questions and type /exit."
  pause
  person login "$brain_missing" "$brain_missing: Claude Code could not be asked"
  ok "signed in"
fi

title "hands' Claude Code plugin"
# This step comes after the sign-in steps, so declining the plugin leaves all earlier steps complete.
note "The plugin lets every Claude Code session tell hands what it is doing."
person install-plugin "hands' Claude Code plugin is not installed, so no session joins hands" \
  "hands' Claude Code plugin is not installed, so no session joins hands: Claude Code could not be asked"
ok "installed"

title "The talk key"
# The Input Monitoring permission for the app this installer runs in, which only the user can give. `hands grant`
# waits for it.
person grant "hands cannot hear the talk key without the Input Monitoring permission" \
  "hands cannot hear the talk key: the Input Monitoring permission cannot be given here"
ok "hands hears Right Shift in every app"

printf '\n%s%sAll set.%s hands %s is installed.\n\n' "$bold" "$green" "$reset" "$version"
# [LAW:one-source-of-truth] hands starts its sessions with the PATH a new terminal has, the same PATH `hands check`
# checks there. It runs the hands this installer installed, regardless of what that PATH finds first.
run_path=$(shell_path "$terminal") || fail "Your shell ($login_shell $terminal) exited without printing its PATH, so hands cannot start sessions with a new terminal's PATH."
note "Starting hands. Hold Right Shift in any app, say what you want, and let go."
note "The first start downloads its speech model, about 1.6 GB. Press q here to quit;"
note "\`hands run\` starts it again. Start Claude Code with \`claude\` in a new terminal."
printf '\n'
ran=0
PATH=$run_path "$bin/hands" run || ran=$?
case $ran in
  0) ;;
  # [LAW:single-enforcer] The lock in hands' home directory determines whether hands is already running. `hands run`
  # exits with code 3 in that case.
  3) note "hands is already running, so the installer did not start a second copy. This terminal is now a login shell with the updated PATH." ;;
  *) fail "hands stopped with exit code $ran. See the message above. \`hands run\` starts it again." ;;
esac
exec "$login_shell" -l
