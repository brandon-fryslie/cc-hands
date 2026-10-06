"""The ways the voice reaches its model, as values: what each needs, and nothing that loads Pipecat to say so."""

from dataclasses import dataclass, field
from pathlib import Path

from hands.core.wire import UPSTREAM


# [LAW:types-are-the-program] the two ways to reach a model differ in what
# they need, not in what they do, so each is a variant with exactly its own
# fields; there is no bag of optional keys and URLs to guard downstream.
@dataclass(frozen=True)
class AnthropicBackend:
    """Claude over the Anthropic API, or any server that speaks it."""

    base_url: str
    # Kept out of the repr, so a backend printed or logged does not print its key.
    api_key: str = field(repr=False)
    model: str


@dataclass(frozen=True)
class OpenAICompatibleBackend:
    """Any OpenAI chat completions server: OpenAI's own API, or one that speaks it."""

    base_url: str
    # Sent as the bearer token on every call, whatever the server does with it; kept out of the repr like Claude's.
    api_key: str = field(repr=False)
    model: str


@dataclass(frozen=True)
class Account:
    """What the brain's requests are billed to: its login, as Claude Code's `auth status` says it."""

    # Claude Code's authMethod: "claude.ai" for a Claude plan, "api_key" for an Anthropic Console key, or another it adds.
    method: str
    # Who holds it: the email a Claude plan is logged in under, or where Claude Code reads the key from.
    holder: str

    def __str__(self) -> str:
        return f"{self.holder} ({self.method})"


@dataclass(frozen=True)
class ClaudeCodeBackend:
    """Claude through a slim Claude Code of hands' own, on whatever login `config_dir` holds (hands.brain)."""

    # No URL of its own: its requests go through hands' proxy, whose address is known only once the run has it listening.
    model: str
    config_dir: Path
    # The account its login held when the run started.
    account: Account


LLMBackend = AnthropicBackend | OpenAICompatibleBackend | ClaudeCodeBackend


def server(backend: LLMBackend) -> str:
    """The server a backend's model answers on: the brain's is Anthropic's, reached through hands' proxy."""
    match backend:
        case AnthropicBackend(base_url=base_url) | OpenAICompatibleBackend(base_url=base_url):
            return base_url
        case ClaudeCodeBackend():
            return UPSTREAM


def account(backend: LLMBackend) -> Account | None:
    """The account the brain runs on, as it was when the run started; None for a keyed variant, whose key is never said."""
    match backend:
        case AnthropicBackend() | OpenAICompatibleBackend():
            return None
        case ClaudeCodeBackend(account=account):
            return account
