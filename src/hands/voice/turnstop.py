"""The end of a user turn: once Whisper is done with every hold of the key the turn took in.

A turn is Pipecat's; a hold is the key's. They are not one to one: Pipecat keeps a turn open until it ends, so a press
while Whisper is still transcribing the last hold joins that turn rather than opening one. Whisper numbers each hold as
it opens, and says when it is done with it, after pushing whatever text the hold had, or at once for a hold the key
dropped. The turn ends when every hold it took in is done, so no hold's words are left out of it, and a hold with no
words ends it as promptly as one with.
"""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, Frame, VADUserStartedSpeakingFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import BaseUserTurnStopStrategy


@dataclass(kw_only=True)
class TurnOpened(VADUserStartedSpeakingFrame):
    """The user started speaking: the key went down, opening the hold with this number."""

    hold: int


@dataclass(kw_only=True)
class TurnResolved(DataFrame):
    """Whisper is done with the hold with this number: whatever text it had has been pushed ahead of this."""

    hold: int


class KeyTurnStop(BaseUserTurnStopStrategy):
    """Ends the user turn when every hold opened in it has been resolved."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  (Pipecat's **kwargs is untyped)
        # [LAW:one-source-of-truth] Whisper opens and resolves the holds; this is the set it has opened and not yet
        # resolved. A set, because a dropped hold is resolved at once, ahead of an earlier one still being transcribed.
        self._open: set[int] = set()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        match frame:
            case TurnOpened(hold=hold):
                self._open.add(hold)
            case TurnResolved(hold=hold):
                # Whisper opens a hold, a system frame, before it pushes anything that resolves it, and a system frame is
                # never overtaken; a hold resolved but never opened is Whisper's bug, so it fails here.
                self._open.remove(hold)
                if not self._open:
                    # A turn whose holds had no text ends with nothing aggregated, so nothing is sent to the model.
                    await self.trigger_user_turn_stopped()
            case _:
                pass
        return ProcessFrameResult.CONTINUE
