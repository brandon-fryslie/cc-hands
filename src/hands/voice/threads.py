"""Blocking work run off the event loop on a thread the process does not wait for."""

import asyncio
import contextlib
import threading
from collections.abc import Callable


async def off_loop[T](work: Callable[[], T], name: str) -> T:
    """The result of work run on a daemon thread, so a process told to stop exits without waiting for it."""
    # asyncio.to_thread's executor thread is joined at exit, which would hold a stopped daemon until its models load.
    loop = asyncio.get_running_loop()
    settled: asyncio.Future[T] = loop.create_future()

    def settle(outcome: Callable[[], None]) -> None:
        if not settled.cancelled():
            outcome()

    def target() -> None:
        try:
            result = work()
        except BaseException as error:
            # Bound now: Python unbinds `error` when the except block ends, before the loop runs the report.
            report: Callable[[], None] = lambda failure=error: settled.set_exception(failure)
        else:
            report = lambda: settled.set_result(result)
        with contextlib.suppress(RuntimeError):
            # The loop is closed only when the run has already ended; there is nobody left to tell.
            loop.call_soon_threadsafe(settle, report)

    threading.Thread(target=target, name=name, daemon=True).start()
    return await settled
