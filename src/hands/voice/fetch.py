"""Models hands hears with, fetched from a release into the home the first time they are needed. Nothing here loads
Pipecat."""

from pathlib import Path

import aiohttp


async def fetched(directory: Path, release: str, names: tuple[str, ...], seconds: float) -> tuple[str, ...]:
    """`names` in `directory`, each fetched from `release` unless already there, and whole or not there at all: the names
    of those fetched. A fetch taking longer than `seconds` in all fails rather than hang whatever waits on it."""
    missing = tuple(name for name in names if not (directory / name).exists())
    directory.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=seconds), raise_for_status=True) as http:
        for name in missing:
            async with http.get(f"{release}/{name}") as response:
                partial = directory / f"{name}.partial"
                partial.write_bytes(await response.read())
            # [LAW:one-source-of-truth] a file under its own name is the whole of it, so one there is never fetched again.
            partial.replace(directory / name)
    return missing
