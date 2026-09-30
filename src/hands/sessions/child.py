"""A child process run to its end inside a time limit, and never left behind by a caller that stops waiting on it."""

import asyncio


async def finished(process: asyncio.subprocess.Process, timeout: float, input: bytes | None = None) -> tuple[bytes, bytes]:
    """What `process` wrote to stdout and stderr once it exits; raises TimeoutError when it runs past `timeout`.

    [LAW:single-enforcer] the one place hands gives up on a child: timed out or its caller cancelled, the process is
    killed and reaped before the error leaves, so the loop never reports a subprocess still running. Each caller says
    what a timeout means for it.
    """
    try:
        return await asyncio.wait_for(process.communicate(input), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
