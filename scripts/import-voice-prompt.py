"""A browser-side Pocket TTS voice, made loadable by the Python package hands speaks with.

Pocket TTS stores a voice two ways. The browser runtime (SlopSpot-paste) keeps the encoded audio prompt: one
`audio_prompt` tensor of [1, frames, 1024]. The Python package keeps the flow model's state after that prompt has
been fed through it: each layer's cache and offset, which `get_state_for_audio_prompt` imports straight from a
.safetensors file. This script feeds the prompt through the model once and writes the state, so a voice made for the
browser is a voice hands can load.

[LAW:one-source-of-truth] the exported state is derived from the prompt by one model, and is garbage under any other:
the output's metadata names the prompt it came from and the config and weights that primed it, and
tests/test_daemon_config.py holds the shipped voice to the weights the package loads today.

Usage: uv run python scripts/import-voice-prompt.py <audio_prompt.safetensors> <out.safetensors>
"""

import sys
from pathlib import Path
from typing import cast

import safetensors.torch
import torch
import yaml
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states

USAGE = "Usage: uv run python scripts/import-voice-prompt.py <audio_prompt.safetensors> <out.safetensors>"

if len(sys.argv) != 3:
    sys.exit(USAGE)
source, out = Path(sys.argv[1]), Path(sys.argv[2])

# Pocket TTS and safetensors ship no types; what each hands back is named here, once, where it comes in.
tensors: dict[str, torch.Tensor] = safetensors.torch.load_file(source)  # pyright: ignore[reportUnknownMemberType]
if set(tensors) != {"audio_prompt"}:
    sys.exit(f"import-voice-prompt: {source} holds {sorted(tensors)}, not the one audio_prompt tensor of a browser voice")

model = TTSModel.load_model()
# [LAW:parse-dont-validate] the prompt is proven to be one batch of frames in the flow model's width before it
# touches the model, so a mis-shaped export is named here and not deep in attention.
raw = tensors["audio_prompt"]
width: int = model.flow_lm.dim
if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != width:
    sys.exit(f"import-voice-prompt: {source} holds an audio_prompt of shape {tuple(raw.shape)}; a voice is [1, frames, {width}]")
prompt = raw.to(device=model.device, dtype=cast(torch.dtype, model.flow_lm.dtype))  # pyright: ignore[reportUnknownMemberType]

with torch.no_grad():
    # The same two steps get_state_for_audio_prompt takes once it has encoded a recording.
    if model.flow_lm.insert_bos_before_voice:
        prompt = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1)
    state = init_states(model.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
    # [LAW:effects-at-boundaries] exception: Pocket TTS has no public call that primes a state from an already
    # encoded prompt; get_state_for_audio_prompt only takes audio or a finished state. This is the one private step.
    model._run_flow_lm_and_increment_step(model_state=state, audio_conditioning=prompt)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]

if model.origin is None:
    sys.exit("import-voice-prompt: the model was not loaded from a packaged config, so the weights cannot be named")
config: dict[str, object] = yaml.safe_load(model.origin.read_text())
provenance = {"source": source.name, "config": model.origin.name, "weights": str(config["weights_path"])}
flat: dict[str, torch.Tensor] = {
    f"{module}/{key}": value.cpu().contiguous() for module, module_state in state.items() for key, value in module_state.items()
}
safetensors.torch.save_file(flat, out, metadata=provenance)  # pyright: ignore[reportUnknownMemberType]
print(f"import-voice-prompt: {out} from {source.name} on {provenance['config']} ({provenance['weights']}), {prompt.shape[1]} frames, {len(flat)} tensors")
