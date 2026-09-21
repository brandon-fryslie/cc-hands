"""A turn, built one record at a time out of the records that make it.

[LAW:one-source-of-truth] the one place a record becomes a step. The tail following a session as it writes and
the backfill reading what a session did before the daemon attached fold the same records the same way, so the
two can never disagree about what a turn did — only about which records they were shown.
"""

from dataclasses import dataclass, field, replace
from typing import cast

from hands.core.steps import Call, Result, recognise
from hands.core.turn import Opening, Said, Step
from hands.sessions.payload import Payload
from hands.sessions.transcript import blocks, holds_a_tool, opening_of, ref_of, result_text, structured_result


@dataclass
class Turning:
    """What has been read of a turn: the steps in the order they happened, and the calls still waiting on results."""

    opening: Opening | None = None
    # A call's result arrives in a later record; its id holds the call's place in the order until it does.
    slots: list[Step | str] = field(default_factory=list[Step | str])
    calls: dict[str, Call] = field(default_factory=dict[str, Call])
    places: dict[str, int] = field(default_factory=dict[str, int])
    mid_tool: bool = False

    def consume(self, record: Payload) -> Opening | None:
        """Read one record in, and say where it opened a turn rather than continuing one.

        The opening is returned rather than taken, because what a new turn means is the reader's to decide: the
        tail lets go of the turn it was following, and a backfill reading a whole morning keeps every one of them.
        """
        opening = opening_of(record, self.mid_tool)
        parts = blocks(record)
        self.mid_tool = holds_a_tool(record)
        if opening is not None:
            # The record that opens a turn is what was asked, not a step of the answer.
            return opening
        ref = ref_of(record)
        # `toolUseResult` describes one call, so a record carrying results for several says which of them it
        # belongs to for none: each is then recognised from its own text, rather than from another call's record.
        structured = structured_result(record) if sum(block.get("type") == "tool_result" for block in parts) == 1 else None
        for block in parts:
            match block:
                case {"type": "text", "text": str() as text} if record.fields.get("type") == "assistant" and text.strip():
                    self.slots.append(Said(ref, text))
                case {"type": "tool_use", "id": str() as id, "name": str() as name, "input": dict()}:
                    held = self.places.get(id)
                    if held is not None and isinstance(self.slots[held], str):
                        # The id has come round again, which a replayed or resumed transcript can do. A result
                        # is paired by id alone, so the call it named before can never be answered now: it is
                        # told as what it was, with no result, rather than borrowing this one's [LAW:no-silent-failure].
                        self.slots[held] = recognise(self.calls[id])
                    self.calls[id] = Call(ref, name, cast(dict[str, object], block["input"]), None)
                    self.places[id] = len(self.slots)
                    self.slots.append(id)
                case {"type": "tool_result", "tool_use_id": str() as id}:
                    place = self.places.get(id)
                    if place is not None:
                        result = Result(result_text(block.get("content")), structured, block.get("is_error") is True)
                        self.slots[place] = recognise(replace(self.calls[id], result=result))
                case _:
                    # Thinking is how Claude reached a result, not a result; the summariser is shown what a turn did.
                    pass
        return None

    def begin(self, opening: Opening) -> None:
        """A turn opened: nothing read of the one before it belongs to this one."""
        self.clear()
        self.opening = opening

    def clear(self) -> None:
        """Let go of the turn read so far, keeping where in a record the reading is."""
        self.opening = None
        self.slots = []
        self.calls = {}
        self.places = {}

    def steps(self) -> list[Step]:
        """Every step read so far. A call whose result has not been written is told as having none."""
        return [slot if not isinstance(slot, str) else recognise(self.calls[slot]) for slot in self.slots]
