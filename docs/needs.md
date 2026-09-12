# What hands-free coding needs

A developer can leave the keyboard for a working session and rely on cc-hands when
nine things are true. Each need below names what breaks when it is missing, the part
of the [architecture](architecture.md) that carries it, and the epic in
[features.md](features.md) that delivers it. The needs are the reason any feature
exists; a feature that serves none of them is not built.

The needs came from one exercise: list every act a developer performs at a keyboard
over a day with coding agents, and ask what has to be true for each act to happen by
voice, without looking, without walking back to the desk. The moment one act cannot,
the developer returns to the keyboard and the product has failed for that moment. So
the bar is not "voice is possible" but "the keyboard is never required."

## 1. Every keyboard act has a spoken equivalent

At the keyboard you prompt, interrupt, answer a question, approve a tool, accept a
plan, pick an option, run `/compact`, start a session in another repo, and close one.
If any of those has no spoken form, the first time you need it you go back to the
desk. Completeness is the need, and it is checked by enumeration: the list of acts is
finite and each one is either reachable or it is a known gap.

Carried by the `Input` union, the `Blocked` state with its three blockers, and the
tool surface. Delivered by the drafts and permissions tickets in the foundation epic,
then **Whole keyboard by voice** and **Sessions by voice**.

## 2. Dictation lands as intended

Speech-to-text turns "auth middleware" into a filename guess and a flag into a word.
The need is not perfect recognition, which does not exist, but that the user hears
what was resolved and can fix it before it goes: the readback speaks what changed,
identifiers are biased toward the words this repo actually uses, a spoken reference
to a file resolves to a real path, and a hard word can be spelled.

Carried by the draft buffer with its stored resolutions and by `find_path`. Delivered
by the drafts ticket and **Dictation fidelity**.

## 3. The right thing is said at the right time

Several sessions share one ear. A permission request from one must come before a
finished turn from another; three turns that finished while you were talking become
one sentence; a request that is still pending is not announced again; nothing starts
while your key is down; a session you have muted stays quiet; and when nothing has
changed, nothing is said. Announce transitions, never states.

Carried by the routing table, the per-session overlay, the priority queue, and the
`coalesce` pass. Delivered by **Attention**.

## 4. Output is understandable by ear, at the depth you choose

A turn's result is a summary by default: "edited three files, tests pass, one
question for you." The details are there on request: what the tests said, the exact
text of a paragraph, a diff read sensibly, a code block skipped with its length named.
"That part" resolves to the record it came from. The summary rests on facts the
daemon computed, not on the agent's own description of what it did.

Carried by the turn ledger and by the `uuid` on every narration. Delivered by
**Narration depth**.

## 5. Nothing happens without you

The intermediary sends only what you approved, approves only what you approved,
denies when you do not answer, and can never edit a file. Every effect it performs is
one line in an audit log, so "did it send something I didn't say" has a definite
answer. The tool surface is small and fixed, and its boundary is names and records,
never file contents.

Carried by the draft buffer, the deny-by-default deadline, the audit log, and the
fixed tool list. Delivered across the foundation epic; the audit log lands in
**Loud daemon**.

## 6. Silence never means broken

In an audio system the failure output and the "still working" output are the same:
nothing. The daemon must be heard or seen to be alive, a hook that cannot reach it
must fail where you can see it, an unreachable model must be spoken, and an error
must reach you by a path that does not run through the thing that broke.

Carried by the system speech channel, the heartbeat file, the non-zero shim exit,
and launchd. Delivered by **Loud daemon**, and it is ranked ahead of every content
feature because transport defects are the ones you feel on the first try (failure
mode 11).

## 7. Fast enough to feel like conversation

The measured spike is 1.4 s from key release to first audio on a plain turn and
4.3 s when a tool call is involved. The need is a bound that holds under load, and
work that skips the model where the model adds nothing: a session finishing is a
template, not a summary; the ledger is computed before the model is asked; the
prompt is cached.

Carried by the `Speak` channel and by the turn slice computed at `Stop`. Delivered
throughout, with the measurement kept in the latency observer.

## 8. It works where you are

At the desk with a headset. Across the room with a hotkey or a button. On the phone
in another room, with the phone's earbuds and its own talk button. Eventually, with
no button at all. Activation without a hand is the last step, not the first, because
an open mic in a room with speakers hears the pipeline's own voice.

Carried by the transport variant and the gate-edge variant. Delivered by
**Presence**.

## 9. It lasts a whole day

The intermediary's own context is the one thing in the system that grows. It must be
compacted without losing the ability to answer "what did we decide this morning," and
a daemon restart must not forget which sessions exist. The long memory is the audit
log; the working memory is the context window; membership is the session files.

Carried by the context summariser, `recall`, `catch_up`, and the session files.
Delivered by **Endurance** and by the restart work in **Loud daemon**.
