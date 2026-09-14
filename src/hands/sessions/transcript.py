"""What a session's JSONL transcript knows that its hooks do not carry."""

from pathlib import Path

from hands.sessions.payload import Payload

# Records are written without spaces. A mention of this text inside a message is
# JSON-escaped, so it cannot match.
_AI_TITLE_RECORD = '"type":"ai-title"'


def ai_title(transcript: Path) -> str | None:
    """The newest title Claude Code gave the session, or None before it has named one."""
    try:
        lines = transcript.open(encoding="utf-8")
    except FileNotFoundError:
        # Claude Code creates the transcript with its first record.
        return None
    title = None
    with lines:
        for line in lines:
            if _AI_TITLE_RECORD in line:
                title = Payload.parse(line.encode()).text("aiTitle")
    return title
