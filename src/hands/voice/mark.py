"""The marks of a user turn: the moments between the key's release and the first sound after it.

One vocabulary for whoever is told of them: the latency log, which says how long after the release each came, and the
phone's page, which shows the user holding it what became of what they said.
"""

from typing import Literal

# In the order a turn that is answered passes them. `released` is the hold taken, to be transcribed and sent, and
# `discarded` the hold thrown away instead; `no words` is a turn Whisper found nothing said in, and `failed` a stage
# failing so that the release goes unanswered, each of which ends the wait. `first audio` is the speaker's first sound
# after the release, whatever hands said with it.
Mark = Literal["released", "discarded", "transcript", "no words", "failed", "first LLM token", "first audio"]
