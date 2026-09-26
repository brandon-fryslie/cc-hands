"""What the intermediary is told when hands starts: which sessions are running, and never what any of them did.

Pointers, not content (docs/architecture.md, 'Push pointers, pull content'): a session's history stays in its
transcript, where read_session pulls it on demand, so the note is the same size after a week of work as after none.
"""

from collections.abc import Mapping, Sequence

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
from pipecat.pipeline.worker import PipelineWorker

from hands.sessions.registry import Sessions
from hands.voice.tools import describe_listing


def briefing(listed: Sequence[Mapping[str, str]]) -> str:
    """The note, from sessions as list_sessions describes them, so the two can never tell the model different things."""
    # [LAW:one-source-of-truth] the id, title, state, and mode are describe_listing's words, the ones list_sessions returns.
    if not listed:
        return "[hands] hands has just started, and no Claude Code sessions are running. Say nothing about this unless the user asks."
    running = "; ".join(_described(session) for session in listed)
    return (
        f"[hands] hands has just started. The Claude Code sessions running now: {running}. "
        "That is how they stood at the start, and they change; list_sessions says how they stand now. "
        "Say nothing about this unless the user asks."
    )


def _described(session: Mapping[str, str]) -> str:
    mode = session["mode"]
    in_mode = "its mode not reported yet" if mode == "not reported yet" else f"in {mode}"
    return f'"{session["title"]}" (id {session["id"]}), {session["state"]}, {in_mode}'


def brief_on_start(worker: PipelineWorker, sessions: Sessions) -> None:
    """Hand the intermediary the note once the pipeline is running, with the model left unrun: it knows, and says nothing."""

    @worker.event_handler("on_pipeline_started")
    async def brief(_worker: PipelineWorker, _frame: Frame) -> None:  # pyright: ignore[reportUnusedFunction]
        # Read at the start rather than at the load: the sweep before the load and the hooks since have both had their say.
        note = briefing([describe_listing(listing) for listing in sessions.live()])
        await worker.queue_frame(LLMMessagesAppendFrame([{"role": "user", "content": note}], run_llm=False))
