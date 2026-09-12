"""Build-order step 1: run the voice pipeline end to end and measure it.

    uv run hands-spike            # Qwen on inferno (HANDS_LLM=local, the default)
    HANDS_LLM=anthropic ANTHROPIC_API_KEY=... uv run hands-spike

Hold the conversation with the space bar: press once to start talking, press
again to stop. `q` quits. Latency from key release to each milestone, ending
with the first audio out of the speaker, is logged for every turn.
"""

import asyncio
import os
import sys

from loguru import logger
from pipecat.services.whisper.stt import MLXModel
from pipecat.workers.runner import WorkerRunner

from hands.voice.keys import drive_key
from hands.voice.pipeline import (
    AnthropicBackend,
    LLMBackend,
    OpenAICompatibleBackend,
    VoiceConfig,
    build_voice,
)
from hands.voice.ptt import Key

# The model lives on inferno, the M4 Max on the LAN, served by mlx_lm.server.
LOCAL_LLM_URL = "http://inferno.local:8080/v1"
LOCAL_LLM_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit"
API_KEY_VAR = "ANTHROPIC_API_KEY"


def backend_from_env() -> LLMBackend:
    """HANDS_LLM picks the variant: `local` (default) or `anthropic`."""
    # [LAW:parse-dont-validate] the environment is parsed here, once, into a
    # variant that carries exactly what its service needs; an unknown choice
    # or a missing key stops the process at the door.
    choice = os.environ.get("HANDS_LLM", "local")
    if choice == "local":
        return OpenAICompatibleBackend(
            base_url=os.environ.get("HANDS_LLM_URL", LOCAL_LLM_URL),
            model=os.environ.get("HANDS_LLM_MODEL", LOCAL_LLM_MODEL),
        )
    if choice == "anthropic":
        api_key = os.environ.get(API_KEY_VAR)
        if not api_key:
            sys.exit(f"{API_KEY_VAR} is not set; HANDS_LLM=anthropic needs it to reach Claude.")
        return AnthropicBackend(
            api_key=api_key,
            model=os.environ.get("HANDS_LLM_MODEL", "claude-haiku-4-5-20251001"),
        )
    sys.exit(f"HANDS_LLM={choice!r} is not one of: local, anthropic.")


def config_from_env() -> VoiceConfig:
    """The process boundary: environment in, typed configuration out."""
    return VoiceConfig(
        llm=backend_from_env(),
        whisper_model=os.environ.get("HANDS_WHISPER_MODEL", MLXModel.LARGE_V3_TURBO),
        voice=os.environ.get("HANDS_VOICE", "alba"),
    )


async def run(config: VoiceConfig) -> None:
    voice = build_voice(config)
    quit_event = asyncio.Event()

    async def on_key(position: Key) -> None:
        turn = voice.key.move_key(position)
        logger.info(f"key {position}: turn {turn}")

    runner = WorkerRunner(handle_sigint=True)
    keys = asyncio.create_task(drive_key(on_key, quit_event))
    logger.info("space: press to talk, press again to stop. q: quit.")
    pipeline_run = asyncio.create_task(runner.run(voice.worker))
    await quit_event.wait()
    await runner.cancel("quit")
    await pipeline_run
    keys.cancel()


def main() -> None:
    asyncio.run(run(config_from_env()))


if __name__ == "__main__":
    main()
