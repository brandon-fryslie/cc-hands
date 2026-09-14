"""The draft tools against a live tmux pane: what is read back is what is typed."""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.services.llm_service import FunctionCallParams

from hands.core.effects import Text
from hands.core.drafts import SendDraft
from hands.core.events import Joined, PermissionRequested
from hands.core.session import Membership, Permission, PromptText, RequestId, SessionId, TmuxPane
from hands.sessions.registry import Sessions
from hands.sessions.tmux import TmuxFailed, type_into
from hands.voice.tools import Tool, draft_tools

RECORDER = Path(__file__).parent / "pane_recorder.py"
WAIT_SECONDS = 5.0


class Pane:
    """A tmux pane running the recorder, and the prompts it has received."""

    def __init__(self, root: Path) -> None:
        self.prompts_file = root / "prompts.json"
        self.name = f"hands-test-{uuid4().hex[:8]}"
        created = subprocess.run(
            ["tmux", "new-session", "-d", "-P", "-F", "#{pane_id}", "-s", self.name, sys.executable, str(RECORDER), str(self.prompts_file)],
            capture_output=True, text=True, check=True,
        )
        self.id = TmuxPane(created.stdout.strip())
        self._wait_for(lambda: self.prompts_file.with_suffix(".ready").exists())

    async def prompts(self, count: int) -> list[str]:
        # Polled without blocking the loop: the send being waited for runs on it.
        deadline = time.monotonic() + WAIT_SECONDS
        while not (self.prompts_file.exists() and len(json.loads(self.prompts_file.read_text())) >= count):
            assert time.monotonic() < deadline, "the pane did not receive the prompts in time"
            await asyncio.sleep(0.02)
        return json.loads(self.prompts_file.read_text())

    def close(self) -> None:
        subprocess.run(["tmux", "kill-session", "-t", self.name], check=True)

    @staticmethod
    def _wait_for(ready: Callable[[], bool]) -> None:
        deadline = time.monotonic() + WAIT_SECONDS
        while not ready():
            assert time.monotonic() < deadline, "the pane did not get there in time"
            time.sleep(0.02)



@pytest.fixture
def pane() -> Iterator[Pane]:
    root = Path(tempfile.mkdtemp(prefix="hands-pane-"))
    live = Pane(root)
    yield live
    live.close()
    shutil.rmtree(root)


async def joined(pane: TmuxPane | None, tmp: Path) -> tuple[Sessions, SessionId]:
    sessions = Sessions(permission_timeout=60.0)
    membership = Membership(SessionId("s1"), pid=1, pane=pane, cwd=Path("/code/cc-hands"), transcript=tmp / "none.jsonl")
    await sessions.apply(Joined(membership, "startup"))
    return sessions, membership.id


async def call(tools: list[Tool], name: str, **arguments: object) -> dict[str, object]:
    results: list[dict[str, object]] = []

    async def capture(result: dict[str, object], **_: object) -> None:
        results.append(result)

    [tool] = [tool for tool in tools if DirectFunctionWrapper(tool).name == name]
    await DirectFunctionWrapper(tool).invoke(arguments, cast(FunctionCallParams, SimpleNamespace(result_callback=capture)))
    [result] = results
    return result


async def test_a_draft_staged_amended_and_sent_lands_in_the_pane_as_read_back(pane: Pane, tmp_path: Path) -> None:
    sessions, id = await joined(pane.id, tmp_path)
    tools = draft_tools(sessions)
    resolved = [{"heard": "auth middleware", "meant": "authMiddleware.ts"}]

    staged = await call(tools, "stage_draft", session=id, text="refactor authMiddleware.ts to use the old helper", resolutions=resolved)
    assert staged == {
        "readback": "Draft for untitled, in cc-hands, reading 'auth middleware' as authMiddleware.ts: "
        "refactor authMiddleware.ts to use the old helper"
    }
    amended = await call(tools, "amend_draft", session=id, text="refactor authMiddleware.ts to use the new token helper", resolutions=resolved)
    assert amended == {"readback": "In the draft for untitled, in cc-hands: 'old' is now 'new token'"}
    assert await call(tools, "send_draft", session=id) == {"readback": "Sent to untitled, in cc-hands."}

    assert await pane.prompts(1) == [" refactor authMiddleware.ts to use the new token helper"]
    assert await call(tools, "send_draft", session=id) == {"readback": "There is no draft for untitled, in cc-hands."}


async def test_a_draft_starting_with_a_sigil_arrives_as_text_with_its_lines_whole(pane: Pane, tmp_path: Path) -> None:
    sessions, id = await joined(pane.id, tmp_path)
    tools = draft_tools(sessions)
    text = "/compact is what I want you to explain\n@README.md second line ends in a backslash \\"
    await call(tools, "stage_draft", session=id, text=text, resolutions=[])
    await call(tools, "send_draft", session=id)
    # The leading space is what keeps Claude Code from reading /compact as the command.
    assert await pane.prompts(1) == [f" {text}"]


async def test_a_session_at_a_permission_dialog_is_sent_nothing_and_keeps_its_draft(pane: Pane, tmp_path: Path) -> None:
    sessions, id = await joined(pane.id, tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="carry on", resolutions=[])
    await sessions.apply(PermissionRequested(id, at=1.0, request=RequestId("r"), permission=Permission("Bash", {})))
    assert await call(tools, "send_draft", session=id) == {
        "readback": "untitled, in cc-hands is waiting for permission to use Bash. Answer that first; the draft is still staged."
    }
    await type_into(pane.id, Text(PromptText("marker")))
    assert await pane.prompts(1) == [" marker"]


@pytest.mark.parametrize(
    ("text", "resolutions", "error"),
    [
        ("  \n", [], "the draft text is empty"),
        ("clear the line\x15and press enter\r", [], "control character"),
        ("fine", [{"heard": "x"}], "missing field 'meant'"),
        ("fine", "none", "resolutions should be a list"),
    ],
)
async def test_arguments_that_do_not_parse_are_refused_out_loud(
    text: str, resolutions: object, error: str, tmp_path: Path
) -> None:
    sessions, id = await joined(TmuxPane("%999999"), tmp_path)
    result = await call(draft_tools(sessions), "stage_draft", session=id, text=text, resolutions=resolutions)
    assert error in str(result["error"])


async def test_a_send_tmux_cannot_type_is_spoken_and_lets_go_of_the_draft(tmp_path: Path) -> None:
    # Keeping it would let the next send type a second copy after whatever reached the pane.
    sessions, id = await joined(TmuxPane("%999999"), tmp_path)
    tools = draft_tools(sessions)
    await call(tools, "stage_draft", session=id, text="hello", resolutions=[])
    result = await call(tools, "send_draft", session=id)
    assert str(result["error"]).startswith("The send failed and the draft is gone")
    assert "can't find pane" in str(result["error"])
    assert await call(tools, "send_draft", session=id) == {"readback": "There is no draft for untitled, in cc-hands."}
    with pytest.raises(TmuxFailed):
        await type_into(TmuxPane("%999999"), Text(PromptText("hello")))


async def test_the_draft_tools_are_valid_pipecat_direct_functions(tmp_path: Path) -> None:
    sessions, _ = await joined(None, tmp_path)
    wrappers = [DirectFunctionWrapper(tool) for tool in draft_tools(sessions)]
    assert [wrapper.name for wrapper in wrappers] == ["stage_draft", "amend_draft", "discard_draft", "send_draft"]
    schema = wrappers[0].to_function_schema()
    assert schema.required == ["session", "text", "resolutions"]
    assert schema.properties["resolutions"]["items"]["required"] == ["heard", "meant"]


async def test_a_machine_without_tmux_is_a_failed_send_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(TmuxFailed, match="cannot run tmux"):
        await type_into(TmuxPane("%1"), Text(PromptText("hello")))


async def test_a_send_cancelled_while_typing_still_reaches_the_pane(pane: Pane, tmp_path: Path) -> None:
    sessions, id = await joined(pane.id, tmp_path)
    await call(draft_tools(sessions), "stage_draft", session=id, text="finish what you started", resolutions=[])
    sending = asyncio.create_task(sessions.draft(SendDraft(id)))
    await asyncio.sleep(0)  # let the send begin before it is cancelled, as a barge-in would
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending
    assert await pane.prompts(1) == [" finish what you started"]


async def test_an_interruption_does_not_cancel_a_draft_tool(tmp_path: Path) -> None:
    sessions, _ = await joined(None, tmp_path)
    assert [getattr(tool, "_pipecat_cancel_on_interruption") for tool in draft_tools(sessions)] == [False] * 4
