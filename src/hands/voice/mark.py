"""The marks of a user turn: the moments between the key's release and the reply being heard.

One vocabulary for whoever is told of them: the latency log, which says how long after the release each came, and the
phone's page, which shows the user holding it what became of what they said.
"""

from typing import Literal

# In the order a turn that is answered passes them. `released` is the hold taken, to be transcribed and sent, and
# `discarded` the hold thrown away instead; `no words` is a hold Whisper found nothing said in, which ends its turn.
Mark = Literal["released", "discarded", "transcript", "no words", "first LLM token", "first audio"]
