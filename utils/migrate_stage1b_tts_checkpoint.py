"""Migrate a legacy Stage-1b checkpoint into canonical PomTTS HF packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from model.constants import SPEECH_VOCAB_SIZE
from model.pom_tts import build_pom_tts


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for one legacy Stage-1b migration run."""
    parser = argparse.ArgumentParser(
        description="Migrate Stage-1b checkpoint keys to canonical PomTTS keys.",
    )
    parser.add_argument(
        "--input-checkpoint-dir",
        required=True,
        help="Path to one legacy checkpoint directory (contains pytorch_model.bin).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to write the migrated HF model package.",
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Write model.safetensors instead of pytorch_model.bin.",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Base model id for PomTTS rebuild (defaults to trainer_state compat or Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--base-cache-dir",
        default=None,
        help="Optional cache directory for loading the base model/tokenizer.",
    )
    parser.add_argument(
        "--speech-vocab-size",
        type=int,
        default=None,
        help="Speech vocab size override (defaults to trainer_state compat or project default).",
    )
    return parser.parse_args()


def _resolve_input_model_path(checkpoint_dir: Path) -> Path:
    """Resolve the model-weight file path from one legacy checkpoint directory."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if bin_path.exists():
        return bin_path
    safe_path = checkpoint_dir / "model.safetensors"
    if safe_path.exists():
        return safe_path
    raise FileNotFoundError(f"no model weights found in {checkpoint_dir}")


def _load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load one state dict from either pytorch_model.bin or model.safetensors."""
    if model_path.suffix == ".bin":
        state = torch.load(model_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"expected state dict mapping in {model_path}")
        return state
    if model_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to read model.safetensors") from exc
        return load_file(str(model_path))
    raise ValueError(f"unsupported model file extension: {model_path.suffix}")


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Strip one leading prefix from every key if all keys contain that prefix."""
    if not state:
        return state
    if not all(key.startswith(prefix) for key in state):
        return state
    return {key[len(prefix):]: value for key, value in state.items()}


def _iter_state_candidates(
    state: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    """Yield deterministic key-normalization candidates for strict loading."""
    seen_signatures: set[Tuple[str, ...]] = set()
    candidates = (
        ("identity", state),
        ("strip_module", _strip_prefix(state, "module.")),
        ("strip_language_model", _strip_prefix(state, "language_model.")),
        ("strip_module_language_model", _strip_prefix(state, "module.language_model.")),
    )
    for name, candidate in candidates:
        signature = tuple(candidate.keys())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        yield name, candidate


def _ensure_safe_output_path(input_dir: Path, output_dir: Path) -> None:
    """Reject output paths that could overwrite or live inside the input checkpoint."""
    in_resolved = input_dir.resolve()
    out_resolved = output_dir.resolve()
    if in_resolved == out_resolved:
        raise ValueError("output-dir must be different from input-checkpoint-dir")
    # Use Path containment semantics so checks are OS/path-separator safe.
    try:
        out_resolved.relative_to(in_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir cannot be inside input-checkpoint-dir")
    if output_dir.exists():
        raise FileExistsError(f"output-dir already exists: {output_dir}")


def _load_trainer_compat(input_dir: Path) -> Dict[str, object]:
    """Load optional trainer compat metadata from one legacy checkpoint directory."""
    trainer_state_path = input_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    try:
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Treat malformed trainer_state as missing compat so caller can enforce explicit provenance.
        return {}
    if not isinstance(state, dict):
        return {}
    compat = state.get("compat")
    if not isinstance(compat, dict):
        return {}
    return compat


def _resolve_build_settings(
    args: argparse.Namespace,
    compat: Dict[str, object],
    *,
    trainer_state_exists: bool,
) -> Tuple[str, int]:
    """Resolve base-model and speech-vocab settings for target PomTTS rebuild."""
    # If trainer_state exists but compat is unavailable, require explicit model provenance.
    if trainer_state_exists and not compat and args.base_model_id is None:
        raise ValueError(
            "trainer_state.json is present but compat metadata is missing/invalid; "
            "pass --base-model-id explicitly"
        )

    base_model_id = (
        str(args.base_model_id)
        if args.base_model_id
        else str(compat.get("model_id", "Qwen/Qwen3-0.6B"))
    )
    speech_vocab_size_raw = args.speech_vocab_size
    if speech_vocab_size_raw is None:
        speech_vocab_size_raw = compat.get("speech_vocab_size", SPEECH_VOCAB_SIZE)
    speech_vocab_size = int(speech_vocab_size_raw)
    if speech_vocab_size <= 0:
        raise ValueError("speech-vocab-size must be > 0")
    return base_model_id, speech_vocab_size


def _try_strict_load_candidate(
    *,
    candidate_state: Dict[str, torch.Tensor],
    base_model_id: str,
    base_cache_dir: str | None,
    speech_vocab_size: int,
):
    """Build a fresh PomTTS and strict-load one candidate state dict into it."""
    model, tokenizer, _ = build_pom_tts(
        base_model_id=base_model_id,
        base_cache_dir=base_cache_dir,
        speech_vocab_size=speech_vocab_size,
    )
    model.load_state_dict(candidate_state, strict=True)
    return model, tokenizer


def _select_strict_candidate(
    *,
    state: Dict[str, torch.Tensor],
    base_model_id: str,
    base_cache_dir: str | None,
    speech_vocab_size: int,
):
    """Pick exactly one key-normalization candidate that strict-loads into PomTTS."""
    success = None
    errors: list[str] = []
    for candidate_name, candidate_state in _iter_state_candidates(state):
        try:
            model, tokenizer = _try_strict_load_candidate(
                candidate_state=candidate_state,
                base_model_id=base_model_id,
                base_cache_dir=base_cache_dir,
                speech_vocab_size=speech_vocab_size,
            )
        except RuntimeError as exc:
            errors.append(f"{candidate_name}: {exc}")
            continue
        if success is not None:
            raise ValueError(
                "multiple key-normalization candidates strict-loaded; "
                "refine candidate rules to keep migration deterministic"
            )
        success = (candidate_name, candidate_state, model, tokenizer)

    if success is None:
        details = "\n".join(errors) if errors else "(no candidates attempted)"
        raise ValueError("no candidate strict-loaded into PomTTS:\n" + details)
    return success


def _assert_single_bin_output(output_dir: Path) -> None:
    """Enforce one-file pytorch_model.bin output in default migration mode."""
    model_bin = output_dir / "pytorch_model.bin"
    shard_bins = sorted(output_dir.glob("pytorch_model-*.bin"))
    shard_index = output_dir / "pytorch_model.bin.index.json"
    safe_file = output_dir / "model.safetensors"
    safe_index = output_dir / "model.safetensors.index.json"
    if not model_bin.exists():
        raise ValueError("default migration output must contain pytorch_model.bin")
    forbidden = [path.name for path in shard_bins]
    if shard_index.exists():
        forbidden.append(shard_index.name)
    if safe_file.exists():
        forbidden.append(safe_file.name)
    if safe_index.exists():
        forbidden.append(safe_index.name)
    if forbidden:
        raise ValueError(
            "default migration output must contain only pytorch_model.bin weights; "
            f"found unsupported files: {', '.join(sorted(forbidden))}"
        )
    # Ensure the default output file is a valid torch state-dict payload.
    payload = torch.load(model_bin, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("default migration pytorch_model.bin must store a state_dict mapping")


def _save_migrated_package(
    *,
    model,
    tokenizer,
    output_dir: Path,
    use_safetensors: bool,
) -> None:
    """Save migrated model/tokenizer with deterministic default .bin behavior."""
    # Always save model config as HF metadata for consistent from_pretrained loading.
    model.config.save_pretrained(output_dir)
    if use_safetensors:
        # Write safetensors weights directly to avoid any implicit save_pretrained behavior.
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required for --safetensors output") from exc
        state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
        save_file(state, str(output_dir / "model.safetensors"))
    else:
        # Default path is explicit torch serialization so output is exactly pytorch_model.bin.
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        _assert_single_bin_output(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Migrate one legacy Stage-1b checkpoint into a HF PomTTS package."""
    args = _parse_args()
    input_dir = Path(args.input_checkpoint_dir)
    output_dir = Path(args.output_dir)
    _ensure_safe_output_path(input_dir, output_dir)

    # Read optional compat metadata so CLI can stay minimal for normal runs.
    compat = _load_trainer_compat(input_dir)
    base_model_id, speech_vocab_size = _resolve_build_settings(
        args,
        compat,
        trainer_state_exists=(input_dir / "trainer_state.json").exists(),
    )

    model_path = _resolve_input_model_path(input_dir)
    state = _load_state_dict(model_path)
    candidate_name, _, model, tokenizer = _select_strict_candidate(
        state=state,
        base_model_id=base_model_id,
        base_cache_dir=args.base_cache_dir,
        speech_vocab_size=speech_vocab_size,
    )

    # Write migration output to a new directory so source checkpoint stays untouched.
    output_dir.mkdir(parents=True, exist_ok=False)
    _save_migrated_package(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        use_safetensors=bool(args.safetensors),
    )

    output_file = "model.safetensors" if args.safetensors else "pytorch_model.bin"
    print(f"input_model: {model_path}")
    print(f"key_migration: {candidate_name}")
    print(f"base_model_id: {base_model_id}")
    print(f"speech_vocab_size: {speech_vocab_size}")
    print(f"output_dir: {output_dir}")
    print(f"output_model_file: {output_file}")


if __name__ == "__main__":
    main()
