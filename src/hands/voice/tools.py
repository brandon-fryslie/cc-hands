"""The tools the intermediary can call. The spike ships one stub."""

from pipecat.services.llm_service import FunctionCallParams


async def list_sessions(params: FunctionCallParams) -> None:
    """List the running Claude Code sessions with their titles.

    Call this when the user asks what is running, what sessions exist, or
    what Claude is working on.
    """
    # Stub for the spike: fixed data so the tool round trip can be measured.
    await params.result_callback(
        {
            "sessions": [
                {"id": "a1", "title": "cc-hands: pipeline spike", "state": "idle"},
                {"id": "b2", "title": "low-talker: keyboard helper", "state": "working"},
            ]
        }
    )
