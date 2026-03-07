"""Export one S2S checkpoint into HF-native Thinker/Talker artifacts.

Canonical usage (from repo root):
  python -m utils.export_hf --input-checkpoint-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from model.pom_talker import build_talker
from model.pom_thinker import build_thinker
from model.tokenizers import ensure_tokenizer_contract


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for one S2S HF export run."""
    parser = argparse.ArgumentParser(description="Export S2S checkpoint to HF artifacts.")
    parser.add_argument(
        "--input-checkpoint-dir",
        required=True,
        help="Path to one S2S checkpoint directory (for example: checkpoints/final).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to write HF output folders: thinker/, talker/, tokenizer/.",
    )
    parser.add_argument(
        "--base-cache-dir",
        default=None,
        help="Optional cache dir for base-model/tokenizer loading.",
    )
    parser.add_argument(
        "--speech-encoder-cache-dir",
        default=None,
        help="Optional cache dir for speech encoder loading during export.",
    )
    return parser.parse_args()


def _resolve_checkpoint_dir(path: str) -> Path:
    """Validate and return one concrete S2S checkpoint directory."""
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def _ensure_safe_output_dir(*, input_dir: Path, output_dir: Path) -> None:
    """Reject output paths that can overwrite the input checkpoint tree."""
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("output-dir must be different from input-checkpoint-dir")
    try:
        output_resolved.relative_to(input_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir cannot be inside input-checkpoint-dir")
    if output_dir.exists():
        raise FileExistsError(f"output-dir already exists: {output_dir}")


def _load_trainer_compat(checkpoint_dir: Path) -> dict[str, Any]:
    """Load required S2S build settings from trainer_state.json compat metadata."""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise FileNotFoundError(f"trainer_state.json not found: {trainer_state_path}")

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"trainer_state.json must be a JSON object: {trainer_state_path}")
    compat = state.get("compat")
    if not isinstance(compat, dict):
        raise ValueError("trainer_state.json is missing compat metadata")

    required = (
        "model_id",
        "speech_encoder_id",
        "frame_stack",
        "adapter_hidden_dim",
        "speech_vocab_size",
    )
    missing = [key for key in required if key not in compat]
    if missing:
        raise ValueError(f"trainer_state compat is missing keys: {missing}")
    return compat


def _resolve_model_files(checkpoint_dir: Path) -> list[Path]:
    """Resolve model checkpoint files and require the 2-file S2S accelerate layout."""
    model_files = sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if len(model_files) != 2:
        names = [path.name for path in model_files]
        raise ValueError(
            "expected exactly two model files in checkpoint dir "
            f"(pytorch_model.bin and one secondary file), found: {names}"
        )
    return model_files


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    """Load one model state_dict and fail fast on invalid payloads."""
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"model file must contain a state_dict mapping: {model_path}")
    return state


def _normalize_state_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip one optional DDP 'module.' prefix from all checkpoint keys."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _build_export_modules(
    *,
    compat: dict[str, Any],
    base_cache_dir: str | None,
    speech_encoder_cache_dir: str | None,
):
    """Build fresh Thinker/Talker plus one shared tokenizer from compat metadata."""
    thinker, tokenizer, token_ids = build_thinker(
        base_model_id=str(compat["model_id"]),
        cache_dir=base_cache_dir,
        speech={
            "encoder_id": str(compat["speech_encoder_id"]),
            "encoder_cache": speech_encoder_cache_dir,
            "frame_stack": int(compat["frame_stack"]),
            "projector_hidden_dim": int(compat["adapter_hidden_dim"]),
        },
    )
    token_ids = ensure_tokenizer_contract(tokenizer, token_ids)
    talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size),
        base_model_id=str(compat["model_id"]),
        base_cache_dir=base_cache_dir,
        speech_vocab_size=int(compat["speech_vocab_size"]),
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    return thinker, talker, tokenizer


def _try_strict_load(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> bool:
    """Return True only when state_dict loads with strict key and shape matching."""
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError:
        return False
    return True


def _classify_states(
    *,
    thinker: torch.nn.Module,
    talker: torch.nn.Module,
    states: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Classify two checkpoint states into thinker/talker by strict-load behavior."""
    thinker_state: dict[str, torch.Tensor] | None = None
    talker_state: dict[str, torch.Tensor] | None = None

    for idx, state in enumerate(states):
        can_load_thinker = _try_strict_load(thinker, state)
        can_load_talker = _try_strict_load(talker, state)

        if can_load_thinker and not can_load_talker:
            if thinker_state is not None:
                raise ValueError("multiple files classified as thinker state")
            thinker_state = state
            continue
        if can_load_talker and not can_load_thinker:
            if talker_state is not None:
                raise ValueError("multiple files classified as talker state")
            talker_state = state
            continue
        raise ValueError(
            f"unable to classify checkpoint file index {idx}: thinker={can_load_thinker}, talker={can_load_talker}"
        )

    if thinker_state is None or talker_state is None:
        raise ValueError("failed to classify thinker and talker checkpoint states")
    return thinker_state, talker_state


def _save_hf_artifacts(
    *,
    thinker: torch.nn.Module,
    talker: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save HF-native artifacts to thinker/, talker/, and tokenizer/ folders."""
    thinker_dir = output_dir / "thinker"
    talker_dir = output_dir / "talker"
    tokenizer_dir = output_dir / "tokenizer"

    output_dir.mkdir(parents=True, exist_ok=False)
    thinker.save_pretrained(thinker_dir, safe_serialization=False)
    talker.save_pretrained(talker_dir, safe_serialization=False)
    tokenizer.save_pretrained(tokenizer_dir)
    return thinker_dir, talker_dir, tokenizer_dir


def _strict_reload_smoke_check(*, thinker_dir: Path, talker_dir: Path) -> None:
    """Reload exported HF artifacts once to verify they are complete and valid."""
    from model.pom_talker import PomTalker
    from model.pom_thinker import PomThinker

    PomThinker.from_pretrained(thinker_dir)
    PomTalker.from_pretrained(talker_dir)


def main() -> None:
    """Run S2S checkpoint export into HF-native model folders."""
    args = _parse_args()
    checkpoint_dir = _resolve_checkpoint_dir(args.input_checkpoint_dir)
    output_dir = Path(args.output_dir)
    _ensure_safe_output_dir(input_dir=checkpoint_dir, output_dir=output_dir)
    compat = _load_trainer_compat(checkpoint_dir)
    model_files = _resolve_model_files(checkpoint_dir)

    # Build clean modules from trainer metadata so export is deterministic.
    thinker, talker, tokenizer = _build_export_modules(
        compat=compat,
        base_cache_dir=args.base_cache_dir,
        speech_encoder_cache_dir=args.speech_encoder_cache_dir,
    )
    states = [_normalize_state_keys(_load_state_dict(path)) for path in model_files]
    thinker_state, talker_state = _classify_states(
        thinker=thinker,
        talker=talker,
        states=states,
    )

    # Strict-load classified states before writing any artifact files.
    thinker.load_state_dict(thinker_state, strict=True)
    talker.load_state_dict(talker_state, strict=True)
    thinker_dir, talker_dir, tokenizer_dir = _save_hf_artifacts(
        thinker=thinker,
        talker=talker,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    _strict_reload_smoke_check(thinker_dir=thinker_dir, talker_dir=talker_dir)

    # Print resolved outputs so inference config wiring is straightforward.
    print(f"input_checkpoint_dir: {checkpoint_dir}")
    print(f"export_thinker_dir: {thinker_dir}")
    print(f"export_talker_dir: {talker_dir}")
    print(f"export_tokenizer_dir: {tokenizer_dir}")


if __name__ == "__main__":
    main()
