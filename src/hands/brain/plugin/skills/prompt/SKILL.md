---
name: prompt
description: How to write a prompt for a Claude Code session from what the user said. Use before every stage_draft and amend_draft.
---

# Writing a prompt for a session

The user speaks roughly; the session reads exactly what you stage. hands reads the draft back to the user before it is
sent, so it is checked by ear: keep it short enough to hear in one go.

The reader is a Claude Code session in the middle of its own work. It knows its code, its task, and what it last said.
It has not heard this conversation. So what it needs from what the user said goes in, and what it already knows stays
out.

Keep:
- Every request, constraint, and decision the user gave, in their own words where the words carry the meaning: "don't
  touch the tests" stays "don't touch the tests".
- Why, when they said why.
- A boundary or a "stop when" they gave for a long piece of work, as the last sentence, where the session looks back to.

Drop:
- Fillers, false starts, and whatever they took back. "The billing one, no, the invoices one" is only "the invoices one".
- The words addressed to you: "tell it to", "ask the docs session".
- Anything they did not say: no added steps, tests, checks, cautions, or "let me know when you're done". The session
  does everything the prompt says, so a line you add is an order the user never gave. When you cannot tell what they
  meant, ask them; don't guess it into the draft.

Write it as the user would type it: plain imperative sentences to the session, no headings or markdown, a list only
when they listed steps. A reply to what the session just asked stays as short as the user said it: "yes, push it" is
the whole prompt. An amendment changes what the user changed and keeps the rest as it was.

The user said: "uh tell billing to, like, fix that rounding thing, but don't change the API, the, no, the public API,
the internal one's fine"
WRONG: "Please investigate the rounding issue in the billing code and add tests for it. Make sure not to change any
APIs. Let me know when you're done!"
RIGHT: "Fix the rounding bug. Don't change the public API; the internal one is fine to change."
