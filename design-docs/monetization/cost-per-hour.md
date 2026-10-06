# What hands costs a user in Anthropic usage

On a day of real use, hands' brain drew about 4% of what the Claude Code sessions it
drove drew from the same Claude plan. At API rates, that was $14.74 over a little more
than seven hours against $373 for the sessions. A user whose plan already carries their
own Claude Code work needs no higher tier for hands. With a keyed intermediary in place
of the brain, hands costs that key about $0.20 for an ordinary hour and at least $1.70
for an hour of steady talk on Sonnet 5.5. Most of the steady hour is a fixed prefix sent
uncached on every request. This settles hands-monetization-76j.lbi.

## How it was measured

hands' audit log records every request the brain and the wrapped sessions make, with
the usage the API reported on its reply (`Exchanged` records in `~/.hands/audit/`).
The log keeps a rolling window, and on 2026-10-06 it held 02:57–10:14 UTC of Brandon's
machine: one brain on its Claude plan as the intermediary (`backend = "claude"`), the
sessions he drove, and the voice turns hands logged (`voice.turn` events). A request
belongs to the brain when its session is one of the brain's transcripts under
`~/.hands/brain/projects/` (its own turns), or a side question the brain asked
(`brain.aside`). Every other request belongs to a driven session. The brain ran on
Claude Sonnet 5. The driven sessions ran on Claude Opus 5.5, apart from 67 requests on
Claude Fable 5.1 and Claude Haiku 4.5. Dollars are first-party API rates as of
2026-09-25, with cache writes and reads priced separately. A Claude plan bills none of
this per token, but its usage limits are spent on the same compute, so the API price is
the common measure.

| Hour (UTC) | Voice turns | Brain requests | Brain tokens | Brain $ | Driven tokens | Driven $ |
|---|---|---|---|---|---|---|
| 03 | 80 | 114 | 23.9M | 9.47 | 113.3M | 48.43 |
| 04 | 8 | 23 | 3.6M | 0.85 | 68.7M | 26.62 |
| 05 | 0 | 19 | 0.1M | 0.09 | 68.1M | 27.53 |
| 06 | 6 | 77 | 2.3M | 1.89 | 101.0M | 43.57 |
| 07 | 2 | 49 | 0.9M | 1.55 | 156.2M | 71.91 |
| 08 | 1 | 69 | 0.7M | 0.31 | 134.5M | 62.91 |
| 09 | 3 | 63 | 1.6M | 0.49 | 147.6M | 68.53 |
| 02:57–02:59, 10:00–10:14 | 0 | 23 | 0.1M | 0.08 | 50.8M | 23.83 |
| **Day** | **100** | **437** | **33.3M** | **14.74** | **840.2M** | **373.33** |

## The brain: what its plan pays for

The brain's cost follows its context size, not the number of words it says. Each of its
own turns re-reads its whole conversation from the prompt cache, and that conversation
held 283,000–326,000 tokens all day. So each turn costs about $0.06 in cache reads
before the brain writes a word. It took 102 turns for 100 voice turns. The 80-turn hour
cost $9.47. Hours with a handful of turns cost $0.08–$1.89.

What a quiet hour costs comes mostly from one rewrite. Claude Code caches the brain's
context for an hour, so the first turn after an hour of silence writes all of it
again, about 322,000 tokens at about $1.30 (06:13 and 07:37 that day). The brain's side
questions, which name sessions and read lines, are separate requests of about 5,600
tokens. They were 335 of its 437 requests, ran 12–71 times an hour, and cost cents.

What a plan tier holds is not published in tokens, so no tier can be computed from
these numbers. The ratio is what a buyer can use: hands exists to drive Claude Code
sessions, so every user already has a plan that carries their sessions, and the brain
adds about 4% to that draw. A user who talks to hands without pause, as in hour 03,
adds about 20% for that hour.

## The keyed intermediary: what a key pays for

The keyed backends (`anthropic` and `openai` in `config.toml`) were not used in the
window, so this part is an estimate built from the measured request rates. A keyed
backend sends two kinds of request to its key. The intermediary's own turns carry its
instruction and its 40 tools' schemas, 35,400 characters, or about 9,000 tokens at four
characters a token. No tokenizer counted them. It takes about one turn per voice turn,
as the brain did, plus one for each telling it says in its own words. The side
questions go to the same key as single summariser calls
(`src/hands/voice/summary.py:26`). Under the brain each carried about 1,250 tokens
beyond the side session's own cached prompt, and a keyed call sends only that part.
The figures below leave out the conversation's history and the tellings, so they are
a floor.

- An hour of steady talk (hour 03: 80 turns, 36 side questions): about 0.77M input
  tokens and 19,000 output tokens, at the 234 tokens a turn the brain averaged. That is
  $1.70 on Sonnet 5.5 or $0.85 on Haiku 4.5.
- An ordinary hour (hours 06–09 averaged: 3 turns, 61 side questions): about 0.1M
  input tokens, or $0.20 on Sonnet 5.5.

An OpenAI-compatible server costs those token counts at its own rates.

The Anthropic backend sends the intermediary's 9,000-token prefix uncached. Pipecat's
Anthropic service leaves prompt caching off unless it is asked for
(`enable_prompt_caching`, default `False`), and `build_llm`
(`src/hands/voice/pipeline.py:151`) does not ask. With caching on, Pipecat marks the
two most recent user messages, so the prefix and the history are both read from the
cache at a tenth of the input price. The steady hour would then cost about $0.40
instead of $1.70.

## What this means for the price of hands

hands' own draw is small next to the sessions it drives. So the cost a buyer weighs
is the plan they already pay for, plus a few dollars of API key a day if they choose
a keyed intermediary. A flat fee for the app does not need to offset either one. The
one cost that grows with use is the brain's context. Each turn pays for the whole
conversation, so keeping that conversation short lowers what every turn costs.
