"""Which writer types into a session, as a table: its membership, the tmux pane it runs in, and the writer. No I/O."""

from dataclasses import replace
from pathlib import Path

import pytest

from hands.core.effects import Fritter
from hands.core.reach import Unwrapped, writer
from hands.core.session import Membership, SessionId
from hands.core.tmux import InPane, NotInTmux, Pane, PaneUnread

SOCKET = Path("/tmp/fritter-1/session.sock")
WRAPPED = Membership(SessionId("s1"), pid=7, cwd=Path("/code/a"), transcript=Path("/t/s1.jsonl"), fritter=SOCKET)
UNWRAPPED = replace(WRAPPED, fritter=None)
PANE = Pane(Path("/tmp/tmux-501/default"), "%3", "work", 1)


@pytest.mark.parametrize("pane", [PANE, NotInTmux(), PaneUnread("tmux at /tmp/tmux-501/default did not answer")])
def test_a_session_fritter_wrapped_is_typed_into_by_its_fritter_wherever_it_runs(pane: InPane) -> None:
    assert writer(WRAPPED, pane) == Fritter(SOCKET, 7)


def test_a_session_nobody_wrapped_is_typed_into_by_tmux_in_its_pane() -> None:
    assert writer(UNWRAPPED, PANE) == PANE


@pytest.mark.parametrize("pane", [NotInTmux(), PaneUnread("tmux at /tmp/tmux-501/default did not answer")])
def test_a_session_with_neither_is_unwrapped_saying_why_no_pane_was_found(pane: NotInTmux | PaneUnread) -> None:
    assert writer(UNWRAPPED, pane) == Unwrapped(UNWRAPPED.id, pane)
