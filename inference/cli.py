"""Tiny CLI entrypoint for Phase A one-turn inference."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from inference.api import PomInferenceAPI
from inference.offline import InferenceResponse


def _parse_args() -> argparse.Namespace:
    """Parse minimal CLI args for one-turn inference."""
    parser = argparse.ArgumentParser(description="Pom Phase A inference CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to canonical inference YAML config",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to one user audio clip",
    )
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--decode-audio",
        dest="decode_audio",
        action="store_true",
        help="Decode generated speech token ids to waveform (default)",
    )
    decode_group.add_argument(
        "--no-decode-audio",
        dest="decode_audio",
        action="store_false",
        help="Skip waveform decode and return only text + speech token ids",
    )
    parser.set_defaults(decode_audio=True)
    parser.add_argument(
        "--text-overrides-json",
        type=partial(_parse_overrides, flag_name="--text-overrides-json"),
        default=None,
        help="Optional JSON object for generation.text overrides",
    )
    parser.add_argument(
        "--speech-overrides-json",
        type=partial(_parse_overrides, flag_name="--speech-overrides-json"),
        default=None,
        help="Optional JSON object for generation.speech overrides",
    )
    return parser.parse_args()


def _parse_overrides(raw: str, *, flag_name: str) -> dict[str, object]:
    """Parse one overrides JSON object from CLI flags."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object")
    return {str(key): value for key, value in parsed.items()}


def _response_to_dict(response: InferenceResponse) -> dict[str, object]:
    """Convert response dataclass to a concise JSON-safe payload."""
    payload: dict[str, object] = {
        "assistant_text": response.assistant_text,
        "assistant_text_token_ids": response.assistant_text_token_ids,
        "speech_token_ids": response.speech_token_ids,
        "thinker_stop_seen": response.thinker_stop_seen,
        "talker_eos_seen": response.talker_eos_seen,
        "conditioning_consumed": response.conditioning_consumed,
        "sample_rate": response.sample_rate,
        "wav_path": response.wav_path,
    }
    # Keep CLI output compact by reporting waveform size instead of raw samples.
    if response.wav is not None:
        payload["wav_num_samples"] = int(response.wav.numel())
    return payload


def main() -> None:
    """Run one CLI inference request through the shared API entrypoint."""
    args = _parse_args()

    # Load runtime once and execute one turn via the same API as Python callers.
    api = PomInferenceAPI.from_config(path=Path(args.config))
    response = api.run_turn(
        audio_path=str(args.audio_path),
        decode_audio=bool(args.decode_audio),
        text_generation_overrides=args.text_overrides_json,
        speech_generation_overrides=args.speech_overrides_json,
    )
    print(json.dumps(_response_to_dict(response), ensure_ascii=True))


if __name__ == "__main__":
    main()
