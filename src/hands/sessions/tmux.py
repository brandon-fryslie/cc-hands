"""The one writer into Claude Code panes."""

import asyncio
from uuid import uuid4

from hands.core.effects import Input, Text
from hands.core.session import TmuxPane


class TmuxFailed(Exception):
    """A tmux command exited non-zero. The message carries the command and its stderr."""


async def type_into(pane: TmuxPane, input: Input) -> None:
    match input:
        case Text(body=body):
            # A buffer of its own per paste, so two sends cannot paste each other's text.
            buffer = f"hands-{uuid4().hex}"
            # [LAW:no-ambient-temporal-coupling] one tmux command list, so the paste and the Enter
            # that submits it happen together or not at all: no half-sent prompt left for a retry to double.
            # Claude Code reads /, @, and ! at the start of the prompt as a command, a file mention,
            # and shell mode. Behind a space each is plain text, and the space goes on every prompt,
            # so nothing inspects the first character. -p brackets the paste, so its newlines stay
            # inside the prompt; -d deletes the buffer once pasted.
            await _tmux(
                ["load-buffer", "-b", buffer, "-", ";", "paste-buffer", "-p", "-d", "-b", buffer, "-t", pane, ";", "send-keys", "-t", pane, "Enter"],
                stdin=f" {body}",
            )


async def _tmux(args: list[str], stdin: str) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            "tmux", *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
    except OSError as error:
        # tmux missing from PATH is a failed send like any other, and is heard as one.
        raise TmuxFailed(f"cannot run tmux: {error}") from error
    _, stderr = await process.communicate(stdin.encode())
    if process.returncode != 0:
        raise TmuxFailed(f"tmux {' '.join(args)} exited {process.returncode}: {stderr.decode(errors='replace').strip()}")
