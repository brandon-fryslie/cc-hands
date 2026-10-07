"""What the model reads beside the user's words, each turn of theirs: what was in front on the Mac's screen as they
spoke, and whether they could see a screen."""

from hands.core import front, place
from hands.core.front import InFront
from hands.core.place import Modality


def beside(in_front: InFront, modality: Modality) -> str:
    """The notes of one turn, as the brain reads them; a screen that could not be read is left out of them."""
    return "\n\n".join(note for note in (front.told(in_front), place.told(modality)) if note)
