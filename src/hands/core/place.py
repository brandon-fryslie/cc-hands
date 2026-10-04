"""Where the user talks to hands from, and whether they can see a screen there."""

from typing import Literal

# The Mac's own microphone and speakers, or the phone's page and its earbuds.
Place = Literal["desk", "phone"]

# Whether the user can see a screen: a hint the brain chooses by, never a limit on what hands does.
Modality = Literal["screen", "audio-only"]


def modality_at(place: Place) -> Modality:
    """The modality a place starts in: the way the user talks to hands sets it, and they can switch it by voice."""
    match place:
        case "desk":
            return "screen"
        case "phone":
            return "audio-only"


def told(modality: Modality) -> str:
    """The note the brain reads beside the user's words."""
    match modality:
        case "screen":
            return "[hands] The user can see a screen."
        case "audio-only":
            return "[hands] The user is audio-only: they cannot see a screen."
