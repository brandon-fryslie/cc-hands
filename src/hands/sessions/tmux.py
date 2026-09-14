"""The one writer into Claude Code panes."""

import subprocess
from uuid import uuid4

from hands.core.effects import Input, Text
from hands.core.session import TmuxPane


class TmuxFailed(Exception):
    """A tmux command exited non-zero. The message carries the command and its stderr."""


def type_into(pane: TmuxPane, input: Input) -> None:
    match input:
        case Text(body=body):
            # Claude Code reads /, @, and ! at the start of the prompt as a command, a file
            # mention, and shell mode. Behind a space each is plain text, and the space goes
            # on every prompt, so nothing inspects the first character.
            _paste(pane, f" {body}")
    _tmux("send-keys", "-t", pane, "Enter")


def _paste(pane: TmuxPane, text: str) -> None:
    # A buffer of its own per paste, so two sends cannot paste each other's text.
    buffer = f"hands-{uuid4().hex}"
    _tmux("load-buffer", "-b", buffer, "-", stdin=text)
    # -p wraps the text in bracketed paste, so its newlines stay inside the prompt instead
    # of submitting it line by line; -d deletes the buffer once pasted.
    _tmux("paste-buffer", "-p", "-d", "-b", buffer, "-t", pane)


def _tmux(*args: str, stdin: str = "") -> None:
    result = subprocess.run(["tmux", *args], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise TmuxFailed(f"tmux {' '.join(args)} exited {result.returncode}: {result.stderr.strip()}")
