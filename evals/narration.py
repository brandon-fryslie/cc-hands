#!/usr/bin/env python3
"""Does a turn's narration say what the turn did, in words that can be heard, in the length it was given?

Run it against whatever model the daemon runs:

    uv run python evals/narration.py                 # the local model on inferno, three tellings each
    uv run python evals/narration.py --runs 1        # one telling each, for a quick look
    HANDS_LLM=anthropic uv run python evals/narration.py

Each case under `evals/fixtures` is a real turn, lifted whole out of a real transcript, beside the facts a
listener has to come away with. The turn is folded by the daemon's own recognisers and rendered by the daemon's
own `render`, so what the model is shown here is byte-for-byte what it is shown at a `Stop` [LAW:one-source-of-truth].

Five checks run on every telling, and a model is stochastic, so every one of them must hold in every run:

  facts    every fact the case demands is in what was said, under one of its wordings.
  spoken   nothing code-shaped reached the ear. The judge is `core.spoken`, the very filter that stands in
           front of the speaker, so this check cannot drift from what the daemon actually does.
  length   what is spoken is no longer than the configured number of sentences, not counting a question, which
           is always said whatever the length. This one guards the code that does the cutting rather than the
           model, which overruns freely; how often it overran is its own line under the summary.
  numbers  every number the model said is a number the turn showed. A local model was heard on 2026-09-21
           reporting "version two point seven point one" for a runner that printed 8.4.1; a number with no
           source in the render is that failure, caught. The headline only: the rest is counted by code.

Exit codes are the contract: 0 every check held, 1 a check failed, 2 the model could not be reached at all.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import anthropic
import openai

from hands.core.delta import Changed, Commit, Delta
from hands.core.narration import Narration, narration, sentences_of
from hands.core.spoken import spoken
from hands.core.turn import Answering, Budget, Continuing, Opening, Step, Turn, render
from hands.daemon.run import SUMMARY_MAX_TOKENS, SUMMARY_TIMEOUT_SECONDS, backend_from_env
from hands.sessions.transcript import turn_record
from hands.sessions.turning import Turning
from hands.voice.narrator import TURN_BUDGET
from hands.voice.summary import SummaryFailed, Summariser, summariser
from hands.voice.summary_instruction import HEADLINE_SENTENCES, TURN_SUMMARY_INSTRUCTION

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Fact:
    """Something the listener must come away knowing, and every wording that would count as saying it."""

    needs: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class Case:
    """One real turn, what git said of it, and what a telling of it has to get right."""

    name: str
    about: str
    turn: Turn
    delta: Delta
    facts: tuple[Fact, ...]
    never: tuple[str, ...]


@dataclass(frozen=True)
class Check:
    """One verdict on one telling. `detail` is what a reader needs to fix it, and is empty when it held."""

    name: str
    held: bool
    detail: str


@dataclass(frozen=True)
class Telling:
    """One run of one case: what was said, how long the model took, and every verdict on it."""

    said: str
    headline: str
    seconds: float
    checks: tuple[Check, ...]
    also_claimed: bool  # the report said what git was about to say, so the listener heard it twice


def cases() -> list[Case]:
    """Every case under `evals/fixtures`, in name order."""
    return [_case(expectation) for expectation in sorted(FIXTURES.glob("*/expect.json"))]


def _case(expectation: Path) -> Case:
    """One case: its expectations, and its transcript folded into the turn the daemon would tell.

    A case may name another's transcript rather than copy it, which is what lets the same turn appear once as a
    first telling and once as a second [LAW:one-source-of-truth].
    """
    written = json.loads(expectation.read_text())
    named = written.get("transcript", "transcript.jsonl")
    opening, steps = _folded(expectation.parent / named if "/" not in named else FIXTURES / named)
    told = int(written.get("told", 0))
    standing = Answering() if told == 0 else Continuing(told)
    return Case(
        name=expectation.parent.name,
        about=written["about"],
        turn=Turn(opening, tuple(steps[told:]), standing),
        delta=_delta(written.get("delta", {})),
        facts=tuple(Fact(tuple(fact["needs"]), fact["why"]) for fact in written["facts"]),
        never=tuple(written.get("never", ())),
    )


def _delta(written: dict[str, object]) -> Delta:
    """What git said the turn left behind, as the real repository recorded it. Empty where a case names none."""
    files = cast(list[list[object]], written.get("files", []))
    commits = cast(list[list[str]], written.get("commits", []))
    return Delta(
        files=tuple(Changed(str(path), cast(int, added), cast(int, removed)) for path, added, removed in files),
        commits=tuple(Commit(sha, subject) for sha, subject in commits),
        patch=str(written.get("patch", "")),
    )


def _folded(transcript: Path) -> tuple[Opening, list[Step]]:
    """The transcript, read into a turn by the recognisers the daemon reads with."""
    turning = Turning()
    for line in transcript.read_bytes().splitlines():
        record = turn_record(line)
        if record is None:
            continue
        opening = turning.consume(record)
        if opening is not None:
            turning.clear()
            turning.begin(opening)
    if turning.opening is None:
        raise SystemExit(f"{transcript} holds no turn: a fixture is a turn and its opening record must be in it.")
    return turning.opening, turning.steps()


async def tell(case: Case, summarise: Summariser, budget: Budget) -> Telling:
    """Summarise the case's turn once and judge what came back."""
    shown = render(case.turn, case.delta, budget)
    began = time.monotonic()
    headline = await summarise(shown)
    seconds = time.monotonic() - began
    told = narration(headline, case.turn, case.delta, HEADLINE_SENTENCES)
    said = told.said()
    checks = (_facts(case, said), _spoken(said), _length(told), _numbers(told.headline.text, shown), _never(case, said))
    return Telling(said, headline, seconds, checks, _also_claimed(told))


def _facts(case: Case, said: str) -> Check:
    """Every fact the case demands, under any of its wordings."""
    heard = said.lower()
    missed = [fact for fact in case.facts if not any(wording.lower() in heard for wording in fact.needs)]
    detail = "; ".join(f"{fact.why} — none of {list(fact.needs)}" for fact in missed)
    return Check("facts", not missed, detail)


def _spoken(said: str) -> Check:
    """Nothing code-shaped reached the ear, judged by the filter that stands in front of the speaker."""
    # [LAW:one-source-of-truth] `core.spoken` is what "code-shaped" means in this daemon. A second table of
    # patterns here would be a rival definition, and the two would disagree the week either one changed.
    heard = spoken(said)
    changed = heard.text != said
    leaked = "; ".join(str(leak) for leak in heard.leaks)
    detail = f"the filter had to rewrite it to {heard.text!r}" if changed else ""
    return Check("spoken", not changed and not heard.leaks, f"{detail}{'; ' if detail and leaked else ''}{leaked}")


def _length(told: Narration) -> Check:
    """What is spoken is within its number of sentences. A question is always said, so it is not counted.

    This judges the narration and not the model, and cannot be read as judging the model: `_headline` cuts to
    the number before this sees it, so the check holds by construction and fails only if that cut regresses,
    which is what it is here to catch. What the model did with the instruction is the `overran` line in `main`,
    counted off the reply as it arrived — and on 2026-09-22 it overran in nine of twelve tellings, which is
    exactly why the number is kept by code and this check guards the code that keeps it.
    """
    reported = [sentence for sentence in sentences_of(told.headline.text) if not sentence.endswith("?")]
    return Check(
        "length",
        len(reported) <= HEADLINE_SENTENCES,
        f"{len(reported)} sentences of report where {HEADLINE_SENTENCES} is the number: {reported}",
    )


def _numbers(headline: str, shown: str) -> Check:
    """Every number the model said is a number the turn showed, in digits or in the word the report used for it.

    The headline alone, because it is the only part a model wrote. The rest of the narration counts things
    itself — "left 6 files different" is arithmetic over the delta, and the render lists those files without
    ever printing their number — so judging the whole of it would fail a true count and blame the model for it.
    """
    invented = [number for number in _said_numbers(headline) if number not in shown and number.lower() not in shown.lower()]
    return Check("numbers", not invented, f"nothing in the turn holds {invented}")


# The claim that the turn did something to the repository, which is git's to make. The bare nouns are not here:
# "the high review of the commit" names a commit the report is about and claims nothing about having made it.
_CLAIMED = re.compile(r"\b(committed|committing|pushe[sd]|pushing|opened a pull request|merged)\b", re.IGNORECASE)


def _also_claimed(told: Narration) -> bool:
    """Whether the report claimed a repository result that git is about to state again.

    Counted, not judged. The instruction asks the model to leave commits alone and mostly it does, but it was
    measured live on 2026-09-22 writing "committed the change" over a narration that then said "It committed".
    Dropping git's clause when the report claims one was considered and refused: a model that claims a commit
    that never landed is exactly the case git's clause exists to contradict, and suppressing it there would turn
    a redundancy into a silent wrong answer [LAW:no-silent-failure]. So the listener hears it twice sometimes,
    and this number is how often. Only the report is read: a closing question may name the commit it asks about.
    """
    reported = " ".join(sentence for sentence in sentences_of(told.headline.text) if not sentence.endswith("?"))
    return bool(told.repository) and bool(_CLAIMED.search(reported))


def _never(case: Case, said: str) -> Check:
    """Nothing the case forbids was said: a code name it must not copy, or a half of the turn already told."""
    heard = said.lower()
    slipped = [forbidden for forbidden in case.never if forbidden.lower() in heard]
    return Check("never", not slipped, f"said what it must not: {slipped}")


_WORDS: dict[str, int] = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        " sixteen seventeen eighteen nineteen twenty".split()
    )
}
_WORDS |= {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_NUMBER = r"(?:\d[\d,]*(?:\.\d+)*|" + "|".join(_WORDS) + r")"
_NUMERIC = re.compile(rf"\b{_NUMBER}\b", re.IGNORECASE)
# A number said as a version: the parts joined by the word the report uses for a dot, and nothing else.
_POINTED = re.compile(rf"\b{_NUMBER}(?:\s+point\s+{_NUMBER})+\b", re.IGNORECASE)


def _said_numbers(said: str) -> list[str]:
    """Every number in the report, as the digits the turn would have shown it in.

    A version read out as words is also put back together — "two point seven point one" becomes "2.7.1" —
    because that is the shape the one fabrication seen in the wild took, and neither half of it is findable
    alone. Only a "point" joins two numbers: "twenty eight out of twenty eight" is two numbers, not 28.28.
    """
    numbers = [_digits(word) for word in _NUMERIC.findall(said.lower())]
    dotted = [".".join(_digits(part) for part in _NUMERIC.findall(run.lower())) for run in _POINTED.findall(said)]
    return numbers + dotted


def _digits(word: str) -> str:
    return str(_WORDS[word]) if word in _WORDS else word.replace(",", "")


async def main() -> int:
    parsed = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parsed.add_argument("--runs", type=int, default=3, help="tellings of each case; a model is stochastic (default 3)")
    parsed.add_argument("--only", default="", help="run only the cases whose name holds this")
    parsed.add_argument("--show", action="store_true", help="print every telling, not only the ones that failed")
    args = parsed.parse_args()

    backend = backend_from_env()
    summarise = summariser(backend, TURN_SUMMARY_INSTRUCTION, SUMMARY_MAX_TOKENS, SUMMARY_TIMEOUT_SECONDS)
    chosen = [case for case in cases() if args.only in case.name]
    if not chosen:
        raise SystemExit(f"no case under {FIXTURES} holds {args.only!r}")
    print(f"{len(chosen)} cases × {args.runs} tellings, headline of {HEADLINE_SENTENCES} sentence(s), on {backend}\n")

    failures = 0
    overran = 0
    doubled = 0
    seconds: list[float] = []
    for case in chosen:
        print(f"── {case.name}: {case.about}")
        for run in range(args.runs):
            try:
                telling = await tell(case, summarise, TURN_BUDGET)
            except (SummaryFailed, openai.OpenAIError, anthropic.AnthropicError) as error:
                # [LAW:no-silent-failure] a model that cannot be reached is not a failing eval, it is no eval.
                print(f"   cannot reach the model: {type(error).__name__}: {error}")
                return 2
            seconds.append(telling.seconds)
            overran += len([sentence for sentence in sentences_of(telling.headline) if not sentence.endswith("?")]) > HEADLINE_SENTENCES
            doubled += telling.also_claimed
            broke = [check for check in telling.checks if not check.held]
            failures += len(broke)
            mark = "ok  " if not broke else "FAIL"
            print(f"   {mark} {telling.seconds:5.2f}s  {telling.said}" if broke or args.show else f"   ok   {telling.seconds:5.2f}s")
            for check in broke:
                print(f"        {check.name}: {check.detail}")
        print()

    print(f"summary in {min(seconds):.2f}–{max(seconds):.2f}s, median {statistics.median(seconds):.2f}s")
    # Not a verdict: the narration cuts an overlong reply, so this is what the number costs, and the signal for
    # changing it. A model that overruns every time is a model being asked for a length it will not write.
    print(f"the model wrote more than {HEADLINE_SENTENCES} sentence(s) in {overran} of {len(chosen) * args.runs} tellings, and was cut")
    print(f"the report also claimed what git then said in {doubled} of {len(chosen) * args.runs} tellings, and was left to")
    print(f"{failures} failed checks over {len(chosen) * args.runs} tellings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
