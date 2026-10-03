#!/usr/bin/env python3
"""Does the intermediary do the right next thing, in words that can be heard, and nothing it was not told to?

Run it against whatever model the daemon runs:

    uv run python evals/intermediary.py                 # Claude, one run each
    uv run python evals/intermediary.py --runs 3        # three runs each, for a model that samples
    HANDS_LLM=openai uv run --env-file .env python evals/intermediary.py

Each case under `evals/conversations` is one decision point: the conversation up to a moment, and what the model
must do next. The model is asked with the daemon's own prompt, its own tool schemas, and its own start-up note, and
the request is built by the daemon's own Pipecat service and adapter, so what it is shown here is what it is
shown in a run [LAW:one-source-of-truth]. The one difference is that it is asked unstreamed: what it answers is the
same, and a proxy that drops streamed tool calls (hands-llm-000.atw) cannot fake a failure.

A case expects exactly one of three things, and every check must hold in every run. The local model is served at
temperature 0, so one run is all it has to say; a backend that samples needs more.

A model may look before it acts, calling list_sessions first. In a run that listing comes straight back and the model
goes on, so here it is answered with the case's own sessions, as list_sessions would describe them, and the model is
asked again; the step it then takes is the one judged, with every word it said on the way, since those reached the
speaker too. How often it looked first is counted as the latency it costs, since every look is one more round trip
before the user hears anything.

  call     the named tool is called, or one of them where a case names several, with the arguments the case names; `mentions` asks that each of an argument's
           listed wordings, any of them, appears in it. Only the tools the case allows may be called beside it.
  reply    words, and no tool: every fact the case needs is said under one of its wordings, nothing it forbids is,
           it is at most the case's number of sentences, no id the conversation holds is said, and nothing
           code-shaped reached the ear, judged by `core.spoken`, the filter in front of the speaker.
  silent   stay_silent is called, and nothing else is said or called.

Exit codes are the contract: 0 every check held, 1 a check failed, 2 the model could not be reached at all. A model
that was reached and gave no answer, an error status or a turn past the timeout, is a failed check: the daemon's
turn fails the same way.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anthropic
import openai
from loguru import logger
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.llm import OpenAILLMService

from hands.core.spoken import spoken
from hands.sessions.home import Home, default_home
from hands.sessions.overlays import Overlays
from hands.daemon.run import backend_from_env
from hands.sessions.registry import Sessions
from hands.sessions.sentences import Sentences
from hands.voice.briefing import briefing
from hands.voice.intermediary_instruction import INTERMEDIARY_INSTRUCTION
from hands.voice.pipeline import AnthropicBackend, ClaudeCodeBackend, LLMBackend, OpenAICompatibleBackend, VoiceConfig, build_llm
from hands.voice.sentences import SummaryStore
from hands.voice.tools import intermediary_tools, pipecat_function

CONVERSATIONS = Path(__file__).parent / "conversations"

# The daemon's own reply budget, read off its configuration rather than copied.
MAX_REPLY_TOKENS = VoiceConfig.__dataclass_fields__["max_reply_tokens"].default

# A voice turn that takes this long is broken whatever it says.
TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class Check:
    name: str
    held: bool
    detail: str


@dataclass(frozen=True)
class Case:
    name: str
    about: str
    messages: list[LLMContextMessage]
    # The sessions running, as list_sessions describes them: what the start-up note says, and what a look returns.
    sessions: list[dict[str, str]]
    expect: dict[str, Any]
    # Every id in the conversation: none of them is ever said aloud.
    ids: frozenset[str]


def cases() -> list[Case]:
    return [_case(path) for path in sorted(CONVERSATIONS.glob("*.json"))]


def _case(path: Path) -> Case:
    written = json.loads(path.read_text())
    sessions = cast(list[dict[str, str]], written["sessions"])
    # The note the daemon hands over at start, rendered by the daemon, so a case cannot hold a stale copy of it.
    messages = [{"role": "user", "content": briefing(sessions)}, *written["messages"]]
    expect = cast(dict[str, Any], written["expect"])
    kinds = {"call", "reply", "silent"} & expect.keys()
    if len(kinds) != 1:
        raise SystemExit(f"{path.name} expects {sorted(kinds) or 'nothing'}; a case expects exactly one of call, reply, silent")
    ids = {session["id"] for session in sessions} | set(cast(list[str], written.get("ids", [])))
    return Case(path.stem, written["about"], cast(list[LLMContextMessage], messages), sessions, expect, frozenset(ids))


# One question to the model: the conversation so far in, what it said and every tool it called out.
Ask = Callable[[list[LLMContextMessage]], Awaitable[tuple[str, tuple[Call, ...]]]]


def asker(backend: LLMBackend) -> Ask:
    """The one place the backend variant is inspected: each asks the way its daemon service would."""
    scratch = Path(tempfile.mkdtemp())
    tools = intermediary_tools(Sessions(permission_deadline=60.0, clock=lambda: 0.0, record=lambda _entry: None), SummaryStore(Sentences(scratch / "sentences.db")), Overlays(Home(scratch)))
    match backend:
        case OpenAICompatibleBackend(base_url=base_url, api_key=api_key):
            service = build_llm(backend, instruction=INTERMEDIARY_INSTRUCTION, max_tokens=MAX_REPLY_TOKENS)
            # build_llm makes this variant's service; the narrowing only tells the checker what the match already did.
            assert isinstance(service, OpenAILLMService)
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS)

            async def from_openai(messages: list[LLMContextMessage]) -> tuple[str, tuple[Call, ...]]:
                context = LLMContext(messages=list(messages), tools=[pipecat_function(tool) for tool in tools])
                invocation = service.get_llm_adapter().get_llm_invocation_params(
                    context, system_instruction=INTERMEDIARY_INSTRUCTION, convert_developer_to_user=not service.supports_developer_role
                )
                # Pipecat returns the request as a bare dict; it is the chat completions call's keyword arguments.
                params = cast(dict[str, Any], service.build_chat_completion_params(invocation))  # pyright: ignore[reportUnknownMemberType]
                params["stream"] = False
                params.pop("stream_options", None)
                completion = cast(ChatCompletion, await client.chat.completions.create(**params))
                message = completion.choices[0].message
                calls = tuple(
                    Call(call.function.name, _arguments(call.function.arguments)) for call in message.tool_calls or () if call.type == "function"
                )
                return message.content or "", calls

            return from_openai
        case AnthropicBackend(base_url=base_url, api_key=api_key, model=model):
            service_ = build_llm(backend, instruction=INTERMEDIARY_INSTRUCTION, max_tokens=MAX_REPLY_TOKENS)
            assert isinstance(service_, AnthropicLLMService)
            client_ = AsyncAnthropic(base_url=base_url, api_key=api_key, max_retries=0, timeout=TIMEOUT_SECONDS)

            async def from_anthropic(messages: list[LLMContextMessage]) -> tuple[str, tuple[Call, ...]]:
                context = LLMContext(messages=list(messages), tools=[pipecat_function(tool) for tool in tools])
                # Pipecat assembles this request inside its streaming call, with no builder to borrow as the OpenAI
                # path does, so these are that assembly's steps the daemon's settings reach, each the service's own:
                # its adapter call, and the thinking it turns off for a Sonnet that would otherwise think.
                params: dict[str, Any] = {"model": model, "max_tokens": MAX_REPLY_TOKENS, **service_._get_llm_invocation_params(context)}  # pyright: ignore[reportPrivateUsage]
                service_._maybe_disable_thinking(params)  # pyright: ignore[reportPrivateUsage]
                reply = cast(BetaMessage, await client_.beta.messages.create(**params, betas=["interleaved-thinking-2025-05-14"]))
                said = " ".join(block.text for block in reply.content if isinstance(block, BetaTextBlock))
                calls = tuple(Call(block.name, block.input) for block in reply.content if isinstance(block, BetaToolUseBlock))
                return said, calls

            return from_anthropic
        case ClaudeCodeBackend():
            # [LAW:no-silent-failure] the eval asks a Pipecat LLM stage, and the brain is a process with none.
            sys.exit("HANDS_LLM=claude has no Pipecat LLM stage for this eval to ask; pick anthropic or openai.")


# How many times a model may look before the step it takes is judged: once is caution, three times is lost.
MOST_LOOKS = 2


async def answer(ask: Ask, case: Case) -> tuple[str, tuple[Call, ...], int]:
    """What the model did at the case's decision point, after any looks it took first, and how many it took.

    What it said is everything it said on the way there: words beside a look reach the speaker as surely as a reply.
    """
    messages: list[LLMContextMessage] = list(case.messages)
    heard: list[str] = []
    looks = 0
    while True:
        said, calls = await ask(messages)
        heard.append(said.strip())
        looking = [call.name for call in calls] == ["list_sessions"] and "list_sessions" not in wanted(case.expect)
        if not looking or looks == MOST_LOOKS:
            return " ".join(part for part in heard if part), calls, looks
        looks += 1
        id = f"look_{looks}"
        messages += [
            cast(LLMContextMessage, {"role": "assistant", "content": said or None, "tool_calls": [{"id": id, "type": "function", "function": {"name": "list_sessions", "arguments": "{}"}}]}),
            cast(LLMContextMessage, {"role": "tool", "tool_call_id": id, "content": json.dumps({"sessions": case.sessions})}),
        ]


def _arguments(text: str) -> dict[str, object]:
    # [LAW:no-silent-failure] arguments that are not a JSON object are a failed call, shown as what they were.
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {"<unparsed>": text}
    return cast(dict[str, object], parsed) if isinstance(parsed, dict) else {"<unparsed>": text}


def judge(case: Case, said: str, calls: tuple[Call, ...]) -> tuple[Check, ...]:
    expect = case.expect
    if "call" in expect:
        return (_called(expect["call"], calls, set(expect.get("allow", ()))), _quiet_ids(case, said), _spoken_check(said))
    if "silent" in expect:
        names = [call.name for call in calls]
        return (Check("silent", names == ["stay_silent"] and not said.strip(), f"called {names} and said {said!r}"),)
    reply = expect["reply"]
    return (
        Check("no call", not calls, f"called {[call.name for call in calls]} where it should only have answered"),
        Check("said", bool(said.strip()), "said nothing at all"),
        _needs(reply.get("needs", []), said),
        _never(reply.get("never", []), said),
        _sentences(said, int(reply.get("sentences", 2))),
        _quiet_ids(case, said),
        _spoken_check(said),
    )


def wanted(expect: dict[str, Any]) -> tuple[str, ...]:
    """The tools a call case accepts: one name, or several where more than one is right."""
    name = expect.get("call", {}).get("name", ())
    return (name,) if isinstance(name, str) else tuple(cast(list[str], name))


def _called(call_case: dict[str, Any], calls: tuple[Call, ...], allowed: set[str]) -> Check:
    names = wanted({"call": call_case})
    matching = [call for call in calls if call.name in names]
    stray = [call.name for call in calls if call.name not in names and call.name not in allowed]
    if not matching:
        return Check("call", False, f"no {' or '.join(names)} call; called {[call.name for call in calls]}")
    call = matching[0]
    repeated = [name for name, times in Counter(call.name for call in matching).items() if times > 1]
    wrong = [f"{key}={call.arguments.get(key)!r}, not {value!r}" for key, value in call_case.get("args", {}).items() if call.arguments.get(key) != value]
    for key, wordings in call_case.get("mentions", {}).items():
        value = str(call.arguments.get(key, "")).lower()
        wrong += [f"{key} holds none of {group}: {value!r}" for group in wordings if not any(word.lower() in value for word in group)]
    if repeated:
        wrong.append(f"called {repeated} more than once, where the second call replaces or repeats the first")
    if stray:
        wrong.append(f"also called {stray}")
    return Check("call", not wrong, "; ".join(wrong))


def _heard(said: str) -> str:
    """What was said, as the wordings are written: a model's curly apostrophe is the same word as a straight one."""
    return said.lower().replace("\u2019", "'")


def _needs(needs: list[list[str]], said: str) -> Check:
    heard = _heard(said)
    missed = [group for group in needs if not any(word.lower() in heard for word in group)]
    return Check("needs", not missed, f"said none of {missed}")


def _never(never: list[str], said: str) -> Check:
    heard = _heard(said)
    slipped = [word for word in never if word.lower() in heard]
    return Check("never", not slipped, f"said what it must not: {slipped}")


# A sentence ends at its stop, a question mark or an exclamation, and the space after it; a version's dots do not end one.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(said: str, most: int) -> Check:
    sentences = [part for part in _SENTENCE_END.split(said.strip()) if part]
    return Check("length", len(sentences) <= most, f"{len(sentences)} sentences where {most} is the most")


def _quiet_ids(case: Case, said: str) -> Check:
    leaked = [id for id in case.ids if id.lower() in said.lower()]
    return Check("no ids", not leaked, f"said {leaked} aloud")


def _spoken_check(said: str) -> Check:
    # [LAW:one-source-of-truth] `core.spoken` is what "code-shaped" means in this daemon, as in evals/narration.py.
    # Space around a reply never reaches the speaker, so only what is between it is judged.
    said = said.strip()
    heard = spoken(said)
    changed = heard.text != said
    leaked = "; ".join(str(leak) for leak in heard.leaks)
    detail = f"the filter had to rewrite it to {heard.text!r}" if changed else ""
    return Check("spoken", not changed and not heard.leaks, f"{detail}{'; ' if detail and leaked else ''}{leaked}")


def _runs(value: str) -> int:
    runs = int(value)
    if runs < 1:
        raise argparse.ArgumentTypeError(f"each case is run at least once, so {runs} runs nothing")
    return runs


async def main() -> int:
    parsed = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parsed.add_argument("--runs", type=_runs, default=1, help="runs of each case; raise it for a model that samples (default 1)")
    parsed.add_argument("--only", default="", help="run only the cases whose name holds this")
    parsed.add_argument("--show", action="store_true", help="print every answer, not only the ones that failed")
    args = parsed.parse_args()
    # Pipecat logs the whole system instruction at DEBUG when the service is built; the eval's output is its verdicts.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    backend = backend_from_env(default_home())
    ask = asker(backend)
    chosen = [case for case in cases() if args.only in case.name]
    if not chosen:
        raise SystemExit(f"no case under {CONVERSATIONS} holds {args.only!r}")
    print(f"{len(chosen)} cases × {args.runs} runs on {backend}\n")

    failures = 0
    looked = 0
    seconds: list[float] = []
    for case in chosen:
        print(f"── {case.name}: {case.about}")
        for _ in range(args.runs):
            began = time.monotonic()
            try:
                said, calls, looks = await answer(ask, case)
                broke = [check for check in judge(case, said, calls) if not check.held]
            except (openai.APITimeoutError, anthropic.APITimeoutError, openai.APIStatusError, anthropic.APIStatusError) as error:
                # Reached, and no answer: the request the daemon would send failed, which is this run failing.
                # Caught before the connection errors, which the timeouts are a kind of.
                said, calls, looks = "", (), 0
                broke = [Check("answered", False, f"{type(error).__name__}: {error}")]
            except (openai.APIConnectionError, anthropic.APIConnectionError) as error:
                # [LAW:no-silent-failure] a model that cannot be reached is not a failing eval, it is no eval.
                print(f"   cannot reach the model: {type(error).__name__}: {error}")
                print(f"{failures} failed checks before it stopped")
                return 2
            took = time.monotonic() - began
            seconds.append(took)
            looked += looks > 0
            failures += len(broke)
            shown = f"{' '.join(f'{call.name}({json.dumps(call.arguments)})' for call in calls)} {said!r}".strip()
            first = f"(looked {looks}×) " if looks else ""
            print(f"   {'ok  ' if not broke else 'FAIL'} {took:5.2f}s  {first}{shown}" if broke or args.show else f"   ok   {took:5.2f}s")
            for check in broke:
                print(f"        {check.name}: {check.detail}")
        print()

    print(f"answered in {min(seconds):.2f}–{max(seconds):.2f}s, median {statistics.median(seconds):.2f}s")
    # Not a verdict: a look is right when the model has no id for the session, and a round trip of silence when it has.
    print(f"the model listed the sessions before acting in {looked} of {len(chosen) * args.runs} runs")
    print(f"{failures} failed checks over {len(chosen) * args.runs} runs")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
