"""The one reader of JSON that crosses into hands: hook payloads, membership files, transcript records."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast

from hands.core.session import SessionId

# Claude Code's session ids are UUIDs. Anything else could name a path outside the sessions directory.
_SESSION_ID = re.compile(r"[A-Za-z0-9-]+")


class Rejected(Exception):
    """Input that does not parse. The message names what was wrong with it."""


@dataclass(frozen=True)
class Payload:
    fields: Mapping[str, object]

    @classmethod
    def parse(cls, raw: bytes) -> Self:
        try:
            value: object = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise Rejected(f"not JSON: {error}") from error
        return cls.of(value, "the JSON")

    def text(self, key: str) -> str:
        match self._field(key):
            case str() as value:
                return value
            case other:
                raise self._wrong(key, "a string", other)

    def optional_text(self, key: str) -> str | None:
        match self.fields.get(key):
            case None:
                return None
            case str() as value:
                return value
            case other:
                raise self._wrong(key, "a string or null", other)

    def integer(self, key: str) -> int:
        match self._field(key):
            case bool() as other:
                raise self._wrong(key, "an integer", other)
            case int() as value:
                return value
            case other:
                raise self._wrong(key, "an integer", other)

    def mapping(self, key: str) -> Mapping[str, object]:
        value = self._field(key)
        match value:
            case dict():
                return cast(dict[str, object], value)
            case _:
                raise self._wrong(key, "an object", value)

    def items(self, key: str) -> list[object]:
        value = self._field(key)
        match value:
            case list():
                return cast(list[object], value)
            case _:
                raise self._wrong(key, "a list", value)

    def optional_items(self, key: str) -> list[object]:
        """A list that may be left out, which is the same as empty."""
        return [] if self.fields.get(key) is None else self.items(key)

    def optional_flag(self, key: str) -> bool:
        """A flag that may be left out, which is the same as false."""
        match self.fields.get(key):
            case None:
                return False
            case bool() as value:
                return value
            case other:
                raise self._wrong(key, "a boolean or null", other)

    @classmethod
    def of(cls, value: object, what: str) -> Self:
        """A JSON object found inside another, such as one entry of a list; `what` names it when it is not one."""
        match value:
            case dict():
                return cls(cast(dict[str, object], value))
            case _:
                raise Rejected(f"{what} should be a JSON object, got {type(value).__name__}")

    def session_id(self) -> SessionId:
        value = self.text("session_id")
        if not _SESSION_ID.fullmatch(value):
            raise Rejected(f"session_id {value!r} is not a Claude Code session id")
        return SessionId(value)

    def _field(self, key: str) -> object:
        try:
            return self.fields[key]
        except KeyError:
            raise Rejected(f"missing field {key!r}") from None

    @staticmethod
    def _wrong(key: str, expected: str, got: object) -> Rejected:
        return Rejected(f"field {key!r} should be {expected}, got {type(got).__name__}")
