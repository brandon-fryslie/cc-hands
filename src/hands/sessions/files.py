"""Files replaced whole: a reader sees the old contents or the new, never half, and a failed write leaves nothing behind."""

import os
import tempfile
from pathlib import Path


def replace_whole(path: Path, text: str, mode: int) -> None:
    """Write text to a file beside path, give it mode, and rename it over path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.chmod(staged, mode)
        os.replace(staged, path)
    except BaseException:
        # [LAW:no-silent-failure] the error still goes up; only the half-made file is taken away.
        Path(staged).unlink(missing_ok=True)
        raise
