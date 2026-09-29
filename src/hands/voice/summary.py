"""The summariser: one stateless call on the configured model, apart from the intermediary's conversation."""

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from anthropic import AsyncAnthropic, Omit, omit
from anthropic.types import TextBlock, ThinkingConfigDisabledParam
from openai import AsyncOpenAI
from pipecat.services.anthropic.llm import _SONNET_THINKS_BY_DEFAULT_FROM, _sonnet_generation  # pyright: ignore[reportPrivateUsage]

from hands.brain.process import environment, workdir
from hands.sessions.payload import Payload, Rejected
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, LLMBackend, OpenAICompatibleBackend

# A rendered turn in, the spoken summary out.
Summariser = Callable[[str], Awaitable[str]]


class SummaryFailed(Exception):
    """The model answered, but with nothing that can be spoken."""


def summariser(backend: LLMBackend, proxy_url: str, instruction: str, max_tokens: int, timeout: float) -> Summariser:
    """The one place the backend variant is inspected for summaries; the Claude Code variant reaches the API through hands' proxy at `proxy_url`."""
    # [LAW:one-type-per-behavior] both backends take the same turn and give the same text; only the client differs.
    # No retries: a summary that fails is said at once, not after a backoff that sounds like nothing happened.
    match backend:
        case OpenAICompatibleBackend(base_url=base_url, api_key=api_key, model=model):
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

            async def from_openai(turn: str) -> str:
                completion = await openai_client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": instruction}, {"role": "user", "content": turn}],
                )
                return _spoken([choice.message.content or "" for choice in completion.choices[:1]])

            return from_openai
        case AnthropicBackend(base_url=base_url, api_key=api_key, model=model):
            anthropic_client = AsyncAnthropic(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

            async def from_anthropic(turn: str) -> str:
                message = await anthropic_client.messages.create(
                    model=model, max_tokens=max_tokens, system=instruction, messages=[{"role": "user", "content": turn}], thinking=thinking(model)
                )
                return _spoken([block.text for block in message.content if isinstance(block, TextBlock)])

            return from_anthropic
        case ClaudeCodeBackend(model=model, config_dir=config_dir):

            async def from_claude_code(turn: str) -> str:
                return _spoken([await once(config_dir, proxy_url, model, instruction, turn, timeout)])

            return from_claude_code


async def once(config_dir: Path, base_url: str, model: str, instruction: str, text: str, timeout: float) -> str:
    """One stateless answer from a slim Claude Code with no tools, on the login in `config_dir`: `text` in, the reply's text out."""
    # [LAW:one-source-of-truth] the same slimming, login, and empty working directory as the brain's, so the two cost and behave alike.
    asked = await asyncio.create_subprocess_exec(
        "claude", "-p", "--output-format", "json", "--model", model, "--system-prompt", instruction,
        "--tools", "", "--strict-mcp-config", "--setting-sources", "user", "--no-session-persistence",
        cwd=workdir(config_dir),
        env=environment(config_dir, base_url, os.environ),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            out, err = await asyncio.wait_for(asked.communicate(text.encode()), timeout)
        except BaseException:
            # A summary given up on, timed out or its run stopping, takes its process with it.
            if asked.returncode is None:
                asked.kill()
            await asked.wait()
            raise
    except TimeoutError:
        raise SummaryFailed(f"claude -p did not answer in {timeout:.0f}s") from None
    try:
        answer = Payload.parse(out)
        if answer.flag("is_error"):
            raise SummaryFailed(f"claude -p failed: {str(answer.fields)[:300]}")
        return answer.text("result")
    except Rejected as error:
        raise SummaryFailed(f"claude -p exited {asked.returncode} with no answer ({error}): {err.decode(errors='replace')[-300:]}") from None


def thinking(model: str) -> ThinkingConfigDisabledParam | Omit:
    """Thinking off for a model that would otherwise think, and left out for the rest, which reject the setting or do not need it."""
    # [LAW:one-source-of-truth] Pipecat's rule for which models think unasked, the one the pipeline's own service applies:
    # a summary that thinks spends its few tokens before it says a word.
    generation = _sonnet_generation(model)
    return {"type": "disabled"} if generation is not None and generation >= _SONNET_THINKS_BY_DEFAULT_FROM else omit


def _spoken(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts).strip()
    if not text:
        raise SummaryFailed("the model returned no summary")
    return text
