"""Stands in for Claude Code in a tmux pane.

    python pane_recorder.py <prompts.json>

It asks the terminal for bracketed paste, as Claude Code does, then records every
prompt submitted with Enter: paste contents kept whole, newlines included. The
prompts file is rewritten after each one; a sibling .ready file says it is listening.
"""

import json
import os
import sys
import tty
from pathlib import Path

PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"


def main(out: Path) -> None:
    stdin = sys.stdin.fileno()
    tty.setraw(stdin)
    os.write(sys.stdout.fileno(), b"\x1b[?2004h")
    out.with_suffix(".ready").touch()
    prompts: list[str] = []
    prompt = bytearray()
    pasting = False
    while byte := os.read(stdin, 1):
        prompt += byte
        if prompt.endswith(PASTE_START):
            del prompt[-len(PASTE_START) :]
            pasting = True
        elif prompt.endswith(PASTE_END):
            del prompt[-len(PASTE_END) :]
            pasting = False
        elif byte == b"\r" and pasting:
            prompt[-1:] = b"\n"
        elif byte == b"\r":
            del prompt[-1:]
            prompts.append(prompt.decode())
            prompt.clear()
            staging = out.with_suffix(".tmp")
            staging.write_text(json.dumps(prompts))
            staging.replace(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
