#!/bin/bash
# install.sh: take a Mac to an installed hands, in one command the README gives:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/brandon-fryslie/cc-hands/master/install.sh)"
#
# It installs what is missing of Claude Code, Homebrew, PortAudio, uv, and the newest released hands, and leaves what
# is there; a run stopped part-way is finished by running it again. It puts each one's directory on the PATH of the
# person's login shell, and ends in a fresh login shell, so this terminal has them too. The only input it asks for is the
# administrator password Homebrew's own install needs, said before it is asked.
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

# The login shell's PATH is the one every new terminal starts with: a directory it lacks gets one line in the profile
# it reads, and one it has is left as it is.
case $(basename "$SHELL") in
  zsh) profile=$HOME/.zprofile ;;
  bash)
    # A bash login shell reads only the first of these that exists, so the line goes in that one; with none, a new
    # .bash_profile.
    profile=$HOME/.bash_profile
    for name in .profile .bash_login .bash_profile; do [ ! -e "$HOME/$name" ] || profile=$HOME/$name; done ;;
  *) profile= ;;
esac
# A new terminal's login shell starts from launchd's PATH, not this run's, which already has every directory. Its PATH is
# the last line it prints, after whatever the profile itself prints.
login_path=$(env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin "$SHELL" -lc 'printf "\n%s" "$PATH"' </dev/null | tail -n 1)
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

say "done: Claude Code, PortAudio, uv and hands $version are installed; this terminal is now a login shell that finds them"
exec "$SHELL" -l
