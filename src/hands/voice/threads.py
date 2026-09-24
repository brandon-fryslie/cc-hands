"""Blocking work run off the event loop on threads the process does not wait for.

asyncio.to_thread's executor threads are joined at exit, so work that never returns — a model still loading, a
write to an audio device that is gone — would hold a stopped daemon open for as long as it hangs.
"""

import asyncio
import contextlib
import queue
import threading
from collections.abc import Callable


async def off_loop[T](work: Callable[[], T], name: str) -> T:
    """The result of work run on a daemon thread of its own."""
    loop = asyncio.get_running_loop()
    settled: asyncio.Future[T] = loop.create_future()
    threading.Thread(target=_settle, args=(loop, settled, work), name=name, daemon=True).start()
    return await settled


class SerialThread:
    """One daemon thread that runs the work it is given one piece at a time, in the order given.

    For work that must never overlap, such as writes to one audio stream: a caller cancelled while its work runs
    does not stop the work, and the next piece still waits for it to finish.
    """

    def __init__(self, name: str) -> None:
        self._queue: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        threading.Thread(target=self._serve, name=name, daemon=True).start()

    def _serve(self) -> None:
        while True:
            self._queue.get()()

    async def run[T](self, work: Callable[[], T]) -> T:
        """The result of work, once everything given before it has finished and it has run."""
        loop = asyncio.get_running_loop()
        settled: asyncio.Future[T] = loop.create_future()
        self._queue.put(lambda: _settle(loop, settled, work))
        return await settled


def _settle[T](loop: asyncio.AbstractEventLoop, settled: asyncio.Future[T], work: Callable[[], T]) -> None:
    """Run work on this thread and settle the future with its outcome, on the loop that awaits it."""
    try:
        result = work()
    except BaseException as error:
        # Bound now: Python unbinds `error` when the except block ends, before the loop runs the report.
        report: Callable[[], None] = lambda failure=error: settled.set_exception(failure)
    else:
        report = lambda: settled.set_result(result)

    def deliver() -> None:
        if not settled.cancelled():
            report()

    with contextlib.suppress(RuntimeError):
        # The loop is closed only when the run has already ended; there is nobody left to tell.
        loop.call_soon_threadsafe(deliver)
