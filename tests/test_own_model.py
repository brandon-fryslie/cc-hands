"""hands' own model: chosen by voice as an edit to config.toml, made only once the user has heard hands say it is switching."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import pytest
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import Running, running
from hands.daemon import config
from hands.daemon.config import Claude, Config, OpenAI
from hands.sessions.audit import Entry, SettingsEdited
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.player import Mark, Marks, Player
from hands.voice.tool import Tool, silent
from hands.sessions.wide import WideEvent
from hands.voice.tools import Called, audited, model_tools
from test_playback import Heard

SPOKEN_FILE = """# hands, by hand
[llm]
backend = "claude"  # the brain
model = "claude-sonnet-5-5"

[telemetry]
collector = "http://otel.example:4318"
"""


def _own(home: Home, reachable: Callable[[Config], object] = lambda _config: None) -> config.OwnModel:
    return config.OwnModel(home, config.load(home), reachable)


def test_a_model_chosen_is_an_edit_to_the_file_that_keeps_every_other_line_as_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    _own(home).weigh("claude-opus-5-5")()
    assert home.config.read_text() == SPOKEN_FILE.replace('model = "claude-sonnet-5-5"', 'model = "claude-opus-5-5"')
    assert config.load(home).config == Config(llm=Claude(model="claude-opus-5-5"), collector="http://otel.example:4318")


def test_a_model_chosen_with_no_file_is_the_default_backend_on_that_model(tmp_path: Path) -> None:
    home = Home(tmp_path)
    _own(home).weigh(" claude-opus-5-5 ")()
    assert home.config.read_text() == '[llm]\nmodel = "claude-opus-5-5"\n'
    assert config.load(home).config == Config(llm=config.Anthropic(model="claude-opus-5-5"))


def test_the_model_running_is_the_one_the_run_started_on_not_the_file_since(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text('[llm]\nbackend = "openai"\n')
    own = _own(home)
    home.config.write_text('[llm]\nbackend = "openai"\nmodel = "gpt-other"\n')
    assert own.running() == OpenAI().model


@pytest.mark.parametrize(
    ("model", "said"),
    [
        ("claude opus", "has a space in it"),
        ("claude-sonnet-5-5", "hands runs on claude-sonnet-5-5 already"),
        ("", "should be a non-empty string"),
        ("claude-opus-9", "hands runs Claude on .*, not claude-opus-9$"),
    ],
)
def test_a_model_the_file_could_not_take_is_refused_and_nothing_is_written(tmp_path: Path, model: str, said: str) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    with pytest.raises(Rejected, match=said):
        _own(home).weigh(model)
    assert home.config.read_text() == SPOKEN_FILE


def test_a_model_named_by_hand_may_be_a_local_servers_path_with_a_space_in_it() -> None:
    assert config.parse('[llm]\nbackend = "openai"\nmodel = "/models/ML Models/qwen3"\n').llm == OpenAI(model="/models/ML Models/qwen3")


def test_a_model_the_file_names_but_the_run_is_not_on_is_refused_naming_both_ways_that_happens(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    own = _own(home)
    # Saved while the run went on: an edit `edited` has yet to take, or refused.
    home.config.write_text(SPOKEN_FILE.replace("claude-sonnet-5-5", "claude-opus-5-5"))
    with pytest.raises(Rejected, match="names claude-opus-5-5 already and hands is not on it yet: an edit it is about to take, or one it refused.*hands restart"):
        own.weigh(" claude-opus-5-5 ")


def test_a_model_a_start_could_not_reach_is_refused_before_it_is_written(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)

    def unreachable(_settings: Config) -> None:
        raise Rejected("the brain is not logged in")

    with pytest.raises(Rejected, match="not logged in"):
        _own(home, unreachable).weigh("claude-opus-5-5")
    assert home.config.read_text() == SPOKEN_FILE


def test_a_save_by_hand_after_the_choice_was_weighed_is_not_written_over(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    keep = _own(home).weigh("claude-opus-5-5")
    home.config.write_text(SPOKEN_FILE + "# saved by hand\n")
    with pytest.raises(Rejected, match="saved after the model was chosen"):
        keep()
    assert home.config.read_text() == SPOKEN_FILE + "# saved by hand\n"


def test_a_file_kept_behind_a_link_is_edited_through_it_and_the_link_stays(tmp_path: Path) -> None:
    kept = tmp_path / "dotfiles" / "hands.toml"
    kept.parent.mkdir()
    kept.write_text(SPOKEN_FILE)
    home = Home(tmp_path / "home")
    home.root.mkdir()
    home.config.symlink_to(kept)
    _own(home).weigh("claude-opus-5-5")()
    assert home.config.is_symlink()
    assert 'model = "claude-opus-5-5"' in kept.read_text()


async def test_a_model_kept_is_an_edit_the_run_starts_again_on(tmp_path: Path) -> None:
    home = Home(tmp_path)
    home.config.write_text(SPOKEN_FILE)
    running_on = config.load(home)
    watching = asyncio.create_task(asyncio.wait_for(config.edited(home, lambda _entry: None, lambda _config: None, running_on, period=0.01), 2.0))
    config.OwnModel(home, running_on, lambda _config: None).weigh("claude-opus-5-5")()
    assert await watching == SettingsEdited(path=str(home.config), refused=None)


class FakeOwn:
    """hands' own model, kept in memory: what was weighed, and what was kept."""

    def __init__(self, refusal: str | None = None) -> None:
        self.refusal = refusal
        self.kept: list[str] = []

    def running(self) -> str:
        return "claude-sonnet-5-5"

    def weigh(self, model: str) -> Callable[[], None]:
        if self.refusal is not None:
            raise Rejected(self.refusal)
        return lambda: self.kept.append(model)


@asynccontextmanager
async def spoken(player: Player, speaker: FrameProcessor | None = None) -> AsyncGenerator[tuple[Running, Heard]]:
    """The player's lines through a speaker and output transport, each reduced to a processor that lets frames through,
    with the marks played behind the output as the daemon stands them."""
    speaker, output, heard = speaker or IdentityFilter(), IdentityFilter(), Heard()
    async with running([player.lines, speaker, output, Marks(), heard], [player.watching(IdentityFilter(), speaker, output, lambda: "held key")]) as run:
        yield run, heard


def _tools(own: FakeOwn, player: Player) -> dict[str, Tool]:
    return {tool.name: tool for tool in model_tools(own, player)}


async def test_a_line_heard_to_its_end_is_said_so() -> None:
    player = Player(lambda _entry: None)
    async with spoken(player):
        assert await asyncio.wait_for(player.heard("Switching."), 2.0) is True


class Unfinished(FrameProcessor):
    """An output transport whose speaker never finishes what it is handed: it passes on every frame but a mark."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        match frame:
            case Mark():
                pass
            case _:
                await self.push_frame(frame, direction)


async def test_a_line_is_heard_once_the_output_passes_its_mark_on_whatever_takes_it_after() -> None:
    player = Player(lambda _entry: None)
    # No Marks behind the output: a barge-in empties its queue, so what counts is the output's push, seen in order.
    speaker, output, heard = IdentityFilter(), IdentityFilter(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(IdentityFilter(), speaker, output, lambda: "held key")]) as run:
        hearing = asyncio.create_task(player.heard("Switching."))
        await heard.until(1, Mark)
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(hearing, 2.0) is True


async def test_a_line_cut_off_by_a_barge_in_is_said_so() -> None:
    player = Player(lambda _entry: None)
    # The mark is never passed on, so the line is never heard to its end: only a barge-in can settle it.
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(IdentityFilter(), speaker, output, lambda: "held key")]) as run:
        hearing = asyncio.create_task(player.heard("Switching."))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        assert await asyncio.wait_for(hearing, 2.0) is False


async def test_use_model_switches_once_the_user_has_heard_it_say_so_and_is_the_whole_reply() -> None:
    own, player = FakeOwn(), Player(lambda _entry: None)
    tools = _tools(own, player)
    async with spoken(player) as (_, heard):
        result = await tools["use_model"].body(model="claude-opus-5-5")
        said = await heard.until(1, TTSSpeakFrame)
    line = "Switching to claude-opus-5-5. I'll be back in a few seconds."
    assert [frame.text for frame in said if isinstance(frame, TTSSpeakFrame)] == [line]
    assert result == {"said": line, "running_on": "claude-sonnet-5-5", "switching_to": "claude-opus-5-5"}
    assert own.kept == ["claude-opus-5-5"]
    assert silent(tools["use_model"], result)


async def test_use_model_refused_says_nothing_and_switches_nothing_and_the_model_answers() -> None:
    own, player = FakeOwn(refusal="the brain is not logged in"), Player(lambda _entry: None)
    tools = _tools(own, player)
    async with spoken(player) as (_, heard):
        result = await tools["use_model"].body(model="claude-opus-5-5")
    assert result == {"error": "the brain is not logged in"}
    assert not [frame for frame in heard.frames if isinstance(frame, TTSSpeakFrame)]
    assert own.kept == []
    assert not silent(tools["use_model"], result)


async def test_use_model_cut_off_before_its_line_is_heard_switches_nothing() -> None:
    own, player = FakeOwn(), Player(lambda _entry: None)
    tools = _tools(own, player)
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(IdentityFilter(), speaker, output, lambda: "held key")]) as run:
        call = asyncio.ensure_future(tools["use_model"].body(model="claude-opus-5-5"))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        result = await asyncio.wait_for(call, 2.0)
    assert result == {"error": "The user spoke over hands saying it would switch to claude-opus-5-5, so it did not switch. Ask whether they still want it."}
    assert own.kept == []


async def test_model_in_use_is_the_model_hands_runs_on() -> None:
    tools = _tools(FakeOwn(), Player(lambda _entry: None))
    assert await tools["model_in_use"].body() == {"running_on": "claude-sonnet-5-5"}


async def test_each_model_tool_call_is_one_event_naming_the_model_run_on_and_the_one_switched_to() -> None:
    recorded: list[Entry] = []
    def audited_tools(player: Player) -> dict[str, Tool]:
        return {tool.name: audited(tool, recorded.append) for tool in model_tools(FakeOwn(), player)}

    tools = audited_tools(player := Player(lambda _entry: None))
    async with spoken(player):
        await tools["model_in_use"].body()
        await tools["use_model"].body(model="claude-opus-5-5")
    # A player of its own: its lines stand in one pipeline.
    tools = audited_tools(player := Player(lambda _entry: None))
    speaker, output, heard = IdentityFilter(), Unfinished(), Heard()
    async with running([player.lines, speaker, output, heard], [player.watching(IdentityFilter(), speaker, output, lambda: "held key")]) as run:
        call = asyncio.ensure_future(tools["use_model"].body(model="claude-opus-5-5"))
        await heard.until(1, TTSSpeakFrame)
        await run.worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(call, 2.0)
    events = [entry for entry in recorded if isinstance(entry, WideEvent)]
    assert [(event.outcome, cast(Called, event.facts["called"]).result) for event in events] == [
        ("ok", {"running_on": "claude-sonnet-5-5"}),
        ("ok", {"said": "Switching to claude-opus-5-5. I'll be back in a few seconds.", "running_on": "claude-sonnet-5-5", "switching_to": "claude-opus-5-5"}),
        ("failed", {"error": "The user spoke over hands saying it would switch to claude-opus-5-5, so it did not switch. Ask whether they still want it."}),
    ]


def test_use_model_runs_on_through_a_barge_in_so_the_model_is_told_it_did_not_switch() -> None:
    assert _tools(FakeOwn(), Player(lambda _entry: None))["use_model"].completes
