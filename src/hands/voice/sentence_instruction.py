"""The instruction for the model that writes the summary store's sentences, several items to a reply.

Its text is part of every sentence's key (see `hands.voice.sentences`), so changing a word here rewrites every
sentence on the next pass, and a sentence written under the old words is never served under the new ones.
"""

SENTENCE_INSTRUCTION = """\
You write one sentence for each item you are given, so that someone who has not read an item knows what it is about. The message is a list of items, each between <item id="..."> and </item>. An item is a ticket or an epic from an issue tracker, or a whole backlog. Its text comes first; an item with parts then lists, between <parts> and </parts>, the sentence already written for each of its parts.

Reply with exactly one line per item, in the form

<id>: <sentence>

using the item's id exactly as given, and nothing else: no heading, no blank commentary, no markdown.

The sentence:
- For an item with no parts: what the work is and why, in plain words, from its text. Leave out ids, file paths, and code names unless the item is about nothing else.
- For an item with parts: what the parts add up to, in one sentence; for a backlog, the main threads of work in it. Do not list the parts one by one.
- At most about thirty words. One sentence, no semicolon chains.
- Say only what the item says. Do not guess at status or progress the text does not state.

Do not write lines like these:
- "**hands-a1b**: Fixes the bug."  (markdown, and says nothing)
- "hands-a1b: This ticket is about the ticket described above."  (says nothing)
- "hands-a1b: Adds the thing; also fixes the other thing; and updates the docs; plus tests."  (a list pretending to be a sentence)

A good line:
hands-a1b: Keeps a spoken summary of each finished turn so a listener can ask what a session did without hearing the whole transcript.
"""
