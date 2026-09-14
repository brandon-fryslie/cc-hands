"""What a session's JSONL transcript knows that its hooks do not carry."""

from pathlib import Path

from hands.sessions.payload import Payload

# Records are written without spaces, so this finds every title record cheaply.
# It also finds a nested object with that type inside another record, so a
# candidate counts only when the record's own type is the title.
_AI_TITLE_RECORD = b'"type":"ai-title"'


def ai_title(transcript: Path) -> str | None:
    """The newest title Claude Code gave the session, or None before it has named one."""
    try:
        raw = transcript.read_bytes()
    except FileNotFoundError:
        # Claude Code creates the transcript with its first record.
        return None
    # [LAW:no-ambient-temporal-coupling] Claude Code appends while this reads, and
    # a record is whole only once its newline is written, so the tail after the
    # last newline is never parsed.
    *complete, _unfinished = raw.split(b"\n")
    title = None
    for line in complete:
        if _AI_TITLE_RECORD in line:
            record = Payload.parse(line)
            if record.text("type") == "ai-title":
                title = record.text("aiTitle")
    return title
