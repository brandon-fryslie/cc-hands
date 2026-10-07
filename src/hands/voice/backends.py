"""How the voice reaches its model, as a value: what it needs, and nothing that loads Pipecat to say so."""

from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class Account:
    """What the brain's requests are billed to: its login, as Claude Code's `auth status` says it."""

    # Claude Code's authMethod: "claude.ai" for a Claude plan, "api_key" for an Anthropic Console key, "oauth_token" for
    # a token, or another it adds.
    method: str
    # Who holds it: the email a Claude plan or a Console login is under, or where Claude Code reads a key from; a token
    # names no holder (2.1.289).
    holder: str | None

    def __str__(self) -> str:
        return f"{self.holder or 'a token'} ({self.method})"


@dataclass(frozen=True)
class ClaudeCodeBackend:
    """Claude through a slim Claude Code of hands' own, on whatever login `config_dir` holds (hands.brain)."""

    # No URL of its own: its requests go through hands' proxy, whose address is known only once the run has it listening.
    model: str
    config_dir: Path
    # The account its login held when the run started.
    account: Account
