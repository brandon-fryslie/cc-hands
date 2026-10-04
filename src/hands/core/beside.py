"""What the model reads beside the user's words, each turn of theirs: what was in front on the Mac's screen as they
spoke, and whether they could see a screen."""

from hands.core import front, place
from hands.core.front import InFront
from hands.core.place import Modality


def beside(in_front: InFront, modality: Modality) -> str:
    """The notes of one turn, as either model reads them; a screen that could not be read is left out of them."""
    # [LAW:one-source-of-truth] composed here for the brain's stage and an API model's context alike, so the two are
    # told the same things in the same words.
    return "\n\n".join(note for note in (front.told(in_front), place.told(modality)) if note)
