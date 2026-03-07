# Phase A Inference Usage

This document shows the two supported entrypoints for Phase A one-turn offline inference.

Why this is structured this way: both CLI and Python use the same code path (`PomInferenceAPI -> run_turn`), so behavior stays consistent.

## Export S2S Checkpoint to HF Artifacts

Use this canonical command from the repo root:

```bash
python -m utils.export_hf \
  --input-checkpoint-dir /workspace/outputs/pom/stage2/checkpoints/final \
  --output-dir /workspace/outputs/pom/final
```

Set `models.cache_dir` to the exported root directory.

The runtime derives all required folders from this one path:
- `<cache_dir>/thinker`
- `<cache_dir>/talker`
- `<cache_dir>/tokenizer`
- `<cache_dir>/decoder`

If you published these assets to Hugging Face under one model repo (for example `nine15pm/pom`), pull them into one local cache root:

```bash
export POM_CACHE_DIR="${POM_CACHE_DIR:-$HOME/.cache/pom}"
hf download nine15pm/pom --repo-type model --local-dir "$POM_CACHE_DIR"
```

Then set:
- `models.cache_dir=$POM_CACHE_DIR`

## CLI example

```bash
python -m inference.cli \
  --config configs/inference.yaml \
  --audio-path /workspace/pom/reference/audio_assets/testprompt1.wav \
  --decode-audio
```

Notes:
- `decode_audio` is the default and primary Phase A path.
- Decoder assets are loaded from `<cache_dir>/decoder` when that folder exists.
- Use `--text-overrides-json` and `--speech-overrides-json` for per-request generation overrides.

## Python example

```python
from inference.api import PomInferenceAPI

api = PomInferenceAPI.from_config("configs/inference.yaml")
response = api.run_turn(
    audio_path="/workspace/pom/reference/audio_assets/testprompt1.wav",
    decode_audio=True,
)

print(response.assistant_text)
print(response.speech_token_ids[:20])
print(response.wav_path, response.sample_rate)
```
