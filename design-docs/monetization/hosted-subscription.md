# A hosted subscription of hands

hands cannot be hosted, and a subscription for it cannot include any Claude usage.
What is left to sell is hands itself: a monthly fee for the software and its updates.
Each user brings their own Anthropic API key and their own Claude Code login, and pays
Anthropic for that usage directly. This settles hands-monetization-76j.twv.

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

Anthropic's terms for products built on Claude Code
([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)) rule out
the other half of the idea. A product "may not pay for, resell, or intermediate Claude
usage on their end users' behalf": each user authenticates with their own API key or
their own Claude plan. Developers also may not "collect, store, or intermediate
Claude.ai credentials or session tokens."

So a fee that bundles model usage, or a hands-run key that users draw on, is not
allowed. That holds for the sessions hands drives, for the intermediary model, and for
the brain. As built, hands already meets these terms. The intermediary uses the user's
own key (`ANTHROPIC_API_KEY` or `HANDS_LLM_ANT_KEY`). The brain is an unmodified
Claude Code that the user signs into through Claude Code's own flow with `hands login`.
fritter wraps Claude Code in a pseudo-terminal and leaves the binary unchanged. A paid
hands has to keep all three true.

One point needs Anthropic's confirmation before hands charges money. fritter's tap
runs each session through a local proxy (`ANTHROPIC_BASE_URL`), which passes every
byte through unchanged and keeps a copy. The proxy runs on the user's machine, and the
traffic is billed to the user, so it neither resells nor bills usage. But "intermediate"
is the terms' own word. Ask Anthropic sales, which the terms name as the contact for
this, before launch.

## What the subscription sells

What users would pay for is not having to do what the README now asks: nine install
steps, several of them in the terminal, plus updates done by hand. A subscription pays
for a signed macOS app that installs those pieces and keeps them current, plus
support.

Pricing follows from this. Hands carries no per-user infrastructure and no model usage,
so the fee does not need a usage component. A flat monthly price is enough. Users still
pay Anthropic separately, and that cost will decide whether hands is worth buying.
hands-monetization-76j.lbi measures it as dollars per hour of use. The brain already
reads its own token usage off the wire (`src/hands/brain/usage.py`), and lbi starts
from that.
