"""A browser-side Pocket TTS voice, made loadable by the Python package hands speaks with.

Pocket TTS stores a voice two ways. The browser runtime (SlopSpot-paste) keeps the encoded audio prompt: one
`audio_prompt` tensor of [1, frames, 1024]. The Python package keeps the flow model's state after that prompt has
been fed through it: each layer's cache and offset, which `get_state_for_audio_prompt` imports straight from a
.safetensors file. This script feeds the prompt through the model once and writes the state, so a voice made for the
browser is a voice hands can load. [LAW:one-source-of-truth] the exported state is derived; the prompt file it came
from is named in the output's metadata.

Usage: uv run python scripts/import-voice-prompt.py <audio_prompt.safetensors> <out.safetensors>
"""

import sys
from pathlib import Path

import safetensors.torch
import torch
from pocket_tts import TTSModel
from pocket_tts.models.tts_model import init_states

source, out = (Path(argument) for argument in sys.argv[1:3])
tensors: dict[str, torch.Tensor] = safetensors.torch.load_file(source)
if set(tensors) != {"audio_prompt"}:
    sys.exit(f"import-voice-prompt: {source} holds {sorted(tensors)}, not the one audio_prompt tensor of a browser voice")

model = TTSModel.load_model()
prompt = tensors["audio_prompt"].to(model.device)
with torch.no_grad():
    # The same two steps get_state_for_audio_prompt takes once it has encoded a recording.
    if model.flow_lm.insert_bos_before_voice:
        prompt = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1)
    state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    # [LAW:effects-at-boundaries] exception: Pocket TTS has no public call that primes a state from an already
    # encoded prompt; get_state_for_audio_prompt only takes audio or a finished state. This is the one private step.
    model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)  # pyright: ignore[reportPrivateUsage]

flat: dict[str, torch.Tensor] = {
    f"{module}/{key}": value.cpu().contiguous() for module, module_state in state.items() for key, value in module_state.items()
}
safetensors.torch.save_file(flat, out, metadata={"source": source.name})
print(f"import-voice-prompt: {out} from {source.name}, {prompt.shape[1]} frames, {len(flat)} tensors")
