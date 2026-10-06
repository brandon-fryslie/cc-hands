# A hosted subscription of hands

hands cannot be hosted, and a subscription for it cannot include any Claude usage.
What is left to sell is hands itself: a monthly fee for the software and its updates.
Each user pays Anthropic directly for the Claude usage hands drives, through their own
Claude plan or API key. hands as built does not yet meet Anthropic's terms for a
product that runs Claude Code, and three parts of it have to change or be cleared with
Anthropic before it charges money. This settles hands-monetization-76j.twv.

## Why nothing can be hosted

Everything hands does happens on the user's Mac ([architecture.md](../../docs/architecture.md#shape)).
It hears a global talk key, opens the local microphone, and types into Claude Code
sessions through fritter's unix sockets. It gets hook events over a unix socket too.
None of that can move to a server, because a server cannot hold the user's keyboard,
microphone, or terminal sessions.

The voice pipeline could in principle move to a server, but there is no reason to.
Whisper and pocket-tts already run on the Apple silicon hands requires, at no cost to
anyone. Hosting them would add a network round trip to every spoken turn and send the
user's voice off their machine, and it would buy nothing in return.

## Why the subscription cannot carry Claude usage

Anthropic's terms for products that run Claude Code
([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)) rule out
the other half of the idea. Such a product "may not pay for, resell, or intermediate
Claude usage on their end users' behalf": each user authenticates with their own API
key or their own Claude plan, and is billed for it directly. So a fee that bundles
model usage, or a hands-run key that users draw on, is not allowed, whether for the
sessions hands drives, the intermediary model, or the brain.

The same page sets the conditions a paid hands has to meet:

- It agrees to Anthropic's [Commercial Terms](https://www.anthropic.com/legal/commercial-terms).
- It runs the Claude Code binary as published, and does not "remove, disable, or
  restrict any authentication method built into it."
- It does not "collect, store, or intermediate Claude.ai credentials or session
  tokens," and does not "route requests through Free, Pro, or Max plan credentials on
  behalf of their users."
- Its name and logo do not use "Claude Code" or "Anthropic." The product is named
  hands, but its repository is named cc-hands.

## What has to change before hands charges money

hands meets the first two conditions only in part. fritter wraps Claude Code in a
pseudo-terminal and leaves the binary unchanged, and the brain signs in through Claude
Code's own flow (`hands login`), on any login Claude Code takes from Anthropic: a Claude
plan, or an Anthropic Console key (`hands login --console`). Only a cloud provider is
refused, because its requests bypass hands' proxy (`logged_in`,
`src/hands/brain/process.py`). Two things do not fit as built:

- **The brain's proxy edits plan-billed requests.** The brain's `ANTHROPIC_BASE_URL`
  is hands' own proxy (`src/hands/brain/process.py:270`), and a brain logged in on a
  Claude plan has its requests billed to that plan. The proxy rewrites request bodies, answers some requests itself
  with a reply written as the model's (`src/hands/sessions/proxy.py:141`), and turns
  some API refusals into its own final 502. That is hands sitting between a
  subscription and its requests, which is closest to what the terms forbid.
- **fritter's tap is an intermediary.** It decrypts each session's traffic to
  Anthropic with a certificate authority hands supplies (`--tap-ca`,
  `src/hands/sessions/wrapper.py:234`) and copies every exchange to hands. It passes
  requests and replies through unchanged, drops credential headers from the copies
  (`fritter/tap.go:69`), and runs on the user's machine, so it neither resells nor
  bills usage and hands never holds the session's token. But it is a TLS
  man-in-the-middle on plan-billed traffic, and "intermediate" is the terms' own word.

A brain on a Console key is behind the proxy on the user's own API account, not on a
plan. Whether the proxy may keep editing a brain on a plan, and whether the tap is
allowed, needs written confirmation from Anthropic sales, which the terms name as the
contact for questions about authentication. [terms-question.md](terms-question.md) is
that question.

## What the subscription sells

What users would pay for is not having to do what the README now asks: nine install
steps, several of them in the terminal, plus updates done by hand. A subscription pays
for a signed macOS app that installs those pieces and keeps them current, plus
support.

Pricing follows from this. hands carries no per-user infrastructure and no model usage,
so the fee does not need a usage component. A flat monthly price is enough. Users still
pay Anthropic separately, and that cost decides whether hands is worth buying. It has
two parts. The brain and the sessions hands drives run on the user's Claude plan, which
is flat-rate, so what they cost is the share of the plan's usage limits a day of hands
spends. The intermediary runs on whichever backend `config.toml`
names: an Anthropic API key (from `ANTHROPIC_API_KEY` or the keychain item
`HANDS_LLM_ANT_KEY`), an OpenAI-compatible key, or the brain's plan. Only the keyed
backends cost per token. [cost-per-hour.md](cost-per-hour.md) measures both parts.
