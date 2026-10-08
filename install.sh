#!/bin/bash
# install.sh: take a Mac to an installed hands, in one command the README gives:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/brandon-fryslie/cc-hands/master/install.sh)"
#
# It installs what is missing of Claude Code, Homebrew, PortAudio, uv, the newest released hands, the claude shim that
# starts every session under fritter, and hands' plugin, and leaves what is there; a run stopped part-way is finished by
# running it again. It puts each one's directory on the PATH of the person's login shell, the shim's first, and ends in a
# fresh login shell, so this terminal has them too. The only input it asks for is the administrator password Homebrew's
# own install needs, what Claude Code asks only once (its theme, its login, and whether to trust the folder `hands smoke`
# runs in), the same for the brain, hands' own Claude Code, and the yes Claude Code asks for to install a plugin by running
# a command, each said before it is asked.
set -euo pipefail

REPO=brandon-fryslie/cc-hands
# [LAW:one-source-of-truth] hatch_build.TAG, the platform tag every release's wheel carries; tests hold the two equal.
WHEEL_TAG=py3-none-macosx_12_0_arm64
# Homebrew's own name for where it lives, which is /opt/homebrew on Apple silicon.
HOMEBREW_PREFIX=${HOMEBREW_PREFIX:-/opt/homebrew}

say() { printf 'hands install: %s\n' "$*"; }
fail() { printf 'hands install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "hands runs on macOS on Apple silicon; this is $(uname -s) $(uname -m)"

# What this run installs is found here, whatever PATH the terminal it runs in has.
export PATH="$HOME/.local/bin:$HOMEBREW_PREFIX/bin:$PATH"

# Claude Code's installer puts its native claude in ~/.local/bin, as `hands check` wants it.
if [ -x "$HOME/.local/bin/claude" ]; then
  say "Claude Code is installed"
else
  say "installing Claude Code with its own installer"
  curl -fsSL https://claude.ai/install.sh | bash
fi

if [ -x "$HOMEBREW_PREFIX/bin/brew" ]; then
  say "Homebrew is installed"
else
  say "installing Homebrew, which needs your administrator password to create $HOMEBREW_PREFIX"
  sudo -v
  # Homebrew asks sudo without a prompt in its non-interactive mode, so the password given above is kept fresh until it
  # is done; its own install of the Command Line Tools can outlast sudo's five minutes. The keeper ends with this run
  # however it ends, a failed or interrupted install included, and is stopped here when Homebrew is in.
  while kill -0 $$ 2>/dev/null && sudo -n -v 2>/dev/null; do sleep 60; done &
  keeper=$!
  # [LAW:no-silent-failure] fetched in an assignment, which set -e stops on, not as an argument, which it does not.
  homebrew_installer=$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)
  NONINTERACTIVE=1 /bin/bash -c "$homebrew_installer"
  kill "$keeper" 2>/dev/null || true
fi

# [LAW:dataflow-not-control-flow] each formula is installed only when missing: `brew install` of an outdated one upgrades it,
# and a second run changes no version.
for formula in portaudio uv; do
  if brew list --formula "$formula" >/dev/null 2>&1; then
    say "$formula is installed"
  else
    say "installing $formula with Homebrew"
    brew install "$formula"
  fi
done

# The newest release is the tag github.com/<repo>/releases/latest redirects to.
latest=$(curl -fsSI -o /dev/null -w '%{redirect_url}' "https://github.com/$REPO/releases/latest")
tag=${latest##*/}
# [LAW:parse-dont-validate] a release tag is vX.Y.Z exactly, as release.yml publishes it.
echo "$tag" | grep -Eqx 'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)' || fail "the newest release of $REPO is not a vX.Y.Z tag: $latest"
version=${tag#v}
bin=$(uv tool dir --bin)
if [ -x "$bin/hands" ] && [ "$("$bin/hands" --version 2>/dev/null)" = "hands $version" ]; then
  say "hands $version, the newest release, is installed"
else
  say "installing hands $version, the newest release"
  release=https://github.com/$REPO/releases/download/$tag
  # A Mac has its own Python, 3.9, which uv would otherwise take; hands needs 3.12, which uv fetches. The constraints are
  # the versions the release was tested on. --reinstall replaces a hands that is there but does not answer its version.
  uv tool install --reinstall --python 3.12 --constraints "$release/constraints.txt" "$release/hands-$version-$WHEEL_TAG.whl"
fi
# Claude Code runs `hands plugin` from this PATH to install the plugin.
export PATH="$bin:$PATH"

# hands' home as hands finds it (hands.sessions.home.default_home): HANDS_HOME, or ~/.hands. Its bin holds the claude
# shim, which hands writes and which says it is installed only when it is the claude PATH finds.
shims=${HANDS_HOME:-$HOME/.hands}/bin
PATH="$shims:$PATH" hands install-fritter

# The login shell's PATH is the one every new terminal starts with: a directory it lacks gets one line in the profile
# it reads, and one it has is left as it is. A new terminal's shell is interactive too, and zsh then reads ~/.zshrc after
# the profile: rc is the file such a shell reads last.
case $(basename "$SHELL") in
  zsh) profile=$HOME/.zprofile rc=$HOME/.zshrc ;;
  bash)
    # A bash login shell reads only the first of these that exists, so the line goes in that one; with none, a new
    # .bash_profile.
    profile=$HOME/.bash_profile
    for name in .profile .bash_login .bash_profile; do [ ! -e "$HOME/$name" ] || profile=$HOME/$name; done
    rc=$profile ;;
  *) profile= rc= ;;
esac
# The PATH the person's shell, started with these flags, ends its startup files with. It starts from launchd's PATH, as a
# new terminal's does, not this run's, which already has every directory, and its PATH is the line marked as it, among
# whatever its startup files and a login shell's logout file print. It fails where the shell ends without saying it.
shell_path() {
  local printed
  printed=$(env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin "$SHELL" "$1" 'printf "\n@hands-path@%s\n" "$PATH"' </dev/null) || return
  printed=$(printf '%s\n' "$printed" | sed -n 's/^@hands-path@//p' | tail -n 1)
  [ -n "$printed" ] && printf '%s' "$printed"
}
login_path=$(shell_path -lc) || fail "your shell, $SHELL -lc, run with no terminal, ended without saying its PATH, so which commands a new terminal finds is unknown"
on_login_path() { case ":$login_path:" in *":$1:"*) return 0 ;; *) return 1 ;; esac; }
lines=()
on_login_path "$HOMEBREW_PREFIX/bin" || lines+=("eval \"\$($HOMEBREW_PREFIX/bin/brew shellenv)\"")
on_login_path "$HOME/.local/bin" || lines+=("export PATH=\"\$HOME/.local/bin:\$PATH\"")
[ "$bin" = "$HOME/.local/bin" ] || on_login_path "$bin" || lines+=("export PATH=\"$bin:\$PATH\"")
if [ ${#lines[@]} -gt 0 ]; then
  [ -n "$profile" ] || fail "your shell, $SHELL, is not zsh or bash; add these lines to its login profile and open a new terminal: ${lines[*]}"
  printf '%s\n' "${lines[@]}" >>"$profile"
  say "added to $profile, for your login shell's PATH: ${lines[*]}"
fi
# The shim runs the claude after it on PATH, so it must be the claude a new terminal finds, not only on its PATH: the
# shim's line goes last in the file read last. A shell whose startup files end it when it has no terminal, as one that
# starts tmux does, cannot be asked what a new terminal finds: the login shell is asked instead, and the line goes in the
# profile it reads.
terminal=-ilc
if ! shell_path -ilc >/dev/null; then
  terminal=-lc rc=$profile
  say "your shell, $SHELL -ilc, run with no terminal, ended without saying its PATH, as one whose startup files start tmux does: which claude a new terminal finds is judged from its login shell alone, which reads $profile"
fi
terminal_claude() { PATH=$(shell_path "$terminal") command -v claude; }
line="export PATH=\"$shims:\$PATH\""
if [ "$(terminal_claude)" != "$shims/claude" ]; then
  [ -n "$rc" ] || fail "your shell, $SHELL, is not zsh or bash; add this line last to the file it reads at start and open a new terminal: $line"
  # A line there already has been read and undone; adding it again would undo nothing more.
  [ -f "$rc" ] && grep -qxF "$line" "$rc" || {
    printf '%s\n' "$line" >>"$rc"
    say "added to $rc, so a new terminal's claude is hands': $line"
  }
  [ "$(terminal_claude)" = "$shims/claude" ] || fail "a new terminal's claude is $(terminal_claude), not hands' $shims/claude, though $rc has $line: a startup file read after it puts another claude first; put that line after this one"
fi

# Claude Code's first-run questions and its login, asked once here, so the session `hands smoke` starts waits on none of them.
first_run=0
hands first-run || first_run=$?
case $first_run in
  0) ;;
  1) fail "Claude Code's first run is not finished, so the session \`hands smoke\` starts would wait on its questions; running this command again asks again" ;;
  *) fail "Claude Code's first run is not finished: Claude Code could not be asked, as said above" ;;
esac

# The brain's login, which only the person can make: a brain that holds one keeps it, and nothing is asked.
login=0
hands login || login=$?
case $login in
  0) ;;
  1) fail "the brain is not logged in, so hands has no model to talk with; running this command again asks again" ;;
  *) fail "the brain is not logged in, so hands has no model to talk with: Claude Code could not be asked, as said above" ;;
esac

# Last, so that declining it leaves every step before it done.
plugin=0
hands install-plugin || plugin=$?
case $plugin in
  0) ;;
  1) fail "hands' Claude Code plugin is not installed, so no session joins hands; running this command again asks again" ;;
  *) fail "hands' Claude Code plugin is not installed, so no session joins hands: Claude Code could not be asked, as said above" ;;
esac

say "done: Claude Code, PortAudio, uv, hands $version, its claude shim and its plugin are installed, Claude Code has been through its first run, and the brain is logged in; this terminal is now a login shell that finds them"
exec "$SHELL" -l
