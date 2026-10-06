# What hands costs a user in Anthropic usage

On a day of real use, hands' brain drew about 4% of what the Claude Code sessions it
drove drew from the same Claude plan. At API rates, that was $14.74 over eight hours
against $373 for the sessions. A user whose plan already carries their own Claude Code
work needs no higher tier for hands. With a keyed intermediary in place of the brain,
hands costs that key roughly $0.20 for an ordinary hour and $3 for an hour of steady
talk on Sonnet. Most of that is a fixed prefix sent uncached on every request. This
settles hands-monetization-76j.lbi.

## How it was measured

hands' audit log records every request the brain and the wrapped sessions make, with
the usage the API reported on its reply (`Exchanged` records in `~/.hands/audit/`).
The window is 2026-10-06 02:00–10:59 UTC on Brandon's machine: one brain on its Claude
plan as the intermediary (`backend = "claude"`), the sessions he drove that day, and
the voice turns hands logged (`voice.turn` events). A request belongs to the brain when
its session is one of the brain's transcripts under `~/.hands/brain/projects/`, or a
side question the brain asked (`brain.aside`). Every other request belongs to a driven
session. Dollars are first-party API rates as of 2026-09-25, with cache writes and
reads priced separately. A Claude plan bills none of this per token, but its usage
limits are spent on the same compute, so the API price is the common measure.

| Hour (UTC) | Voice turns | Brain requests | Brain tokens | Brain $ | Driven tokens | Driven $ |
|---|---|---|---|---|---|---|
| 03 | 80 | 114 | 23.9M | 9.47 | 113.3M | 48.43 |
| 04 | 8 | 23 | 3.6M | 0.85 | 68.7M | 26.62 |
| 05 | 0 | 19 | 0.1M | 0.09 | 68.1M | 27.53 |
| 06 | 6 | 77 | 2.3M | 1.89 | 101.0M | 43.57 |
| 07 | 2 | 49 | 0.9M | 1.55 | 156.2M | 71.91 |
| 08 | 1 | 69 | 0.7M | 0.31 | 134.5M | 62.91 |
| 09 | 3 | 63 | 1.6M | 0.49 | 147.6M | 68.53 |
| 02, 10 | 0 | 23 | 0.1M | 0.08 | 50.8M | 23.83 |
| **Day** | **100** | **437** | **33.3M** | **14.74** | **840.2M** | **373.33** |

## The brain: what its plan pays for

The brain's cost follows its context size, not the number of words it says. Every
request re-reads the brain's whole conversation from the prompt cache, and that
conversation held 283,000–326,000 tokens all day. So each request costs about $0.06
in cache reads before the brain writes a word. The 80-turn hour cost $9.47. Hours with
a handful of turns cost $0.08–$1.89.

What a quiet hour costs comes mostly from one rewrite. Claude Code caches the brain's
context for an hour, so the first turn after an hour of silence writes all of it
again, about 322,000 tokens at about $1.30 (06:13 and 07:37 that day). The brain's side
questions, which name sessions and read lines, ran 12–71 times an hour at about 5,000
tokens each, and cost cents.

What a plan tier holds is not published in tokens, so no tier can be computed from
these numbers. The ratio is what a buyer can use: hands exists to drive Claude Code
sessions, so every user already has a plan that carries their sessions, and the brain
adds about 4% to that draw. A user who talks to hands without pause, as in hour 03,
adds about 20% for that hour.

## The keyed intermediary: what a key pays for

The keyed backends (`anthropic` and `openai` in `config.toml`) were not used in the
window, so this part is an estimate built from the measured turn rate. Every request
carries the intermediary's instruction and its 40 tools' schemas, 35,400 characters,
or about 9,000 tokens at four characters a token. No tokenizer counted them. The
conversation's history comes on top of that prefix, so the figures below are a floor.
The brain made about 1.4 requests per voice turn, and a keyed intermediary also makes
one request for each telling it says in its own words.

- An hour of steady talk (80 turns, about 120 requests): about 1.4M input tokens.
  That is $2.90 on Sonnet 5.5 or $1.40 on Haiku 4.5, plus about $0.25 of output
  at the 175 tokens a reply the brain averaged.
- An ordinary hour (a few turns and tellings, about 10 requests): about $0.20 on
  Sonnet 5.5.

An OpenAI-compatible server costs those token counts at its own rates.

The Anthropic backend sends that 9,000-token prefix uncached. Pipecat's Anthropic
service leaves prompt caching off unless it is asked for
(`enable_prompt_caching`, default `False`), and `build_llm`
(`src/hands/voice/pipeline.py:152`) does not ask. With caching on, the prefix would be
read at a tenth of the input price, and the steady hour would cost about $0.50
instead of $3.

## What this means for the price of hands

hands' own draw is small next to the sessions it drives. So the cost a buyer weighs
is the plan they already pay for, plus a few dollars of API key a day if they choose
a keyed intermediary. A flat fee for the app does not need to offset either one. The
one cost that grows with use is the brain's context. Each request pays for the whole
conversation, so keeping that conversation short lowers what every turn costs.
