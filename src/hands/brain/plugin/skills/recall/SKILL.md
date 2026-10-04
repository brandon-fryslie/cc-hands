---
name: recall
description: How to find what was said, sent, or decided earlier. Load it whenever the user asks about anything from before that may have left your context - what was decided, said, or sent to a session, what was allowed or approved, what happened this morning, earlier, or yesterday.
---

# Recalling what was said and sent

Everything said, sent, and approved lately is in hands' audit log, and the recall command your
instructions name reads it in a moment. What you remember of this conversation is only part of it, so look before you
answer, and look even when you think you remember.

Search with the words most likely to be in the moment itself: `<recall command> token helper` prints, oldest first, the
newest moments whose heading or text holds every word. Fewer, more distinctive words find more. When nothing comes
back, no moment held every word: drop a word or cut one to its stem ("token" for "token helper", "deploy" for
"deployed"), because what was sent may spell it differently from how it was said. No words prints the newest moments of
all, for "what did we do this morning". `-n 50` reaches further back than the newest 20. A word that starts with a dash
goes after `--`: `<recall command> -- --no-verify`.

The first line says how far back the log reaches; nothing older can be found. Each line after it is
`<time> <heading>: <text>`:
- `user` is what the user said. It is only what was said: it may have changed before anything was sent.
- `you` is what you said aloud, drafts read back included.
- `sent to <session>` is what was typed into that session: a prompt, a `/command`, or the escape key.
- `not sent to <session>` is a send that never reached it, and why.
- `allowed`, `denied`, `answered`, or `approved in <session>` is how that session's permission request, question, or
  plan was answered; `answered` says what the user chose.

A decision is usually the latest `sent to` or `user` line on it; a later line overrides an earlier one. Answer from the
lines, the answer first, with when and where it came from, in one or two sentences. Say the session's name; a session shown
by its id has no name, so say "a session with no name" rather than read the id. When nothing turns up after widening, say you found nothing about it since the time the first line gives.

The user asked: "what did we decide about the token helper?"
WRONG: "I don't have that in my context anymore." Or: "You sent 4b2e9c1 drop the token helper."
RIGHT: "Drop it. At 9:15 this morning you sent billing: drop the token helper and read the token from the keychain."
