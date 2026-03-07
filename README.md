<p align="center">
  <img src="images/pom.png" alt="Pomegranate (Pom) banner" width="720" />
</p>

# Pomegranate 1B

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/nine15pm/pom)

Pomegranate (Pom) is a tiny ~1B speech-to-speech model with a thinker-talker architecture. It uses a pretrained Qwen 3 0.6B as the LLM backbone, Whisper as the speech encoder, and CosyVoice as the speech decoder.

1. The encoder extracts features from user speech
2. An adapter maps Whisper features into the Thinker's embedding
3. Thinker produces text and hidden states
4. Talker is conditioned on the Thinker's text and hidden states and generates speech tokens with a fixed text<>speech interleaving ratio
5. The decoder uses a flow matching model to decode the speech tokens to mels and vocoder outputs the waveforms

Trained with ~1200 hrs of audio data on a single H100 for ~47 GPU hours.

## Setup

Install inference dependencies only and get model assets

```bash
uv sync
uv run hf download nine15pm/pom --local-dir ~/.cache/pom
```

Install optional training dependencies

```bash
uv sync --extra train 
```

## Test the model

Run the streaming server, then open `http://<server-host>:8000/`

```bash
uv run python -m inference.server --config configs/streaming.yaml
```

Quick usage notes:
1. Accept browser mic permissions.
2. Click and hold `Hold To Talk` while speaking. Release to send one turn.
3. Watch text stream into the transcript and hear audio chunks as they arrive.
4. Click `New Conversation` to clear conversation state.

## Acknowledgements

Built using:
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (`Apache-2.0`)
- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (`Apache-2.0`)
- [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) (`Apache-2.0`)
- [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (`MIT`)
