"""Evaluate TTS (PomTTS) overfit checkpoints on the exact TTS data shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import yaml

# Add repo root so local imports work when script is run from utils/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.constants import IGNORE_INDEX, SPEECH_VOCAB_SIZE
from model.pom_tts import build_pom_tts
from model.tokenizers import TokenIds
from train.tts_sequence_builder import build_read_write_sequence


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tiny TTS overfit evaluation."""
    parser = argparse.ArgumentParser(description="TTS overfit evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to TTS YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint directory or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to write per-sample eval outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples for quick smoke checks",
    )
    return parser.parse_args()


def _get_cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Read one config section and default to an empty mapping."""
    section = cfg.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return section


def _load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config into a top-level dictionary."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping at the top level")
    return data


def _resolve_checkpoint_dir(cfg: Dict[str, Any], checkpoint: str) -> Path:
    """Resolve checkpoint input to a concrete step directory."""
    if checkpoint != "latest":
        path = Path(checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    # Read training.output_dir when --checkpoint=latest.
    train_cfg = _get_cfg_section(cfg, "training")
    output_dir = train_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("training.output_dir must be set when --checkpoint=latest")
    checkpoints_root = Path(str(output_dir)) / "checkpoints"
    candidates = sorted(path for path in checkpoints_root.glob("step-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no step-* checkpoints found under: {checkpoints_root}")
    return candidates[-1]


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON object records from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {lineno}: expected JSON object")
            yield record


def _iter_tts_records(json_path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate TTS records from a shard directory or a single JSONL file."""
    if json_path.is_file():
        yield from _iter_jsonl_records(json_path)
        return
    if not json_path.is_dir():
        raise FileNotFoundError(f"data.json_path not found: {json_path}")

    shard_paths = sorted(json_path.glob("shard-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard-*.jsonl files found in: {json_path}")
    for shard_path in shard_paths:
        yield from _iter_jsonl_records(shard_path)


def _load_eval_samples(cfg: Dict[str, Any], *, limit: Optional[int]) -> list[Dict[str, Any]]:
    """Load and validate TTS eval samples from configured data.json_path."""
    data_cfg = _get_cfg_section(cfg, "data")
    raw_json_path = data_cfg.get("json_path")
    if not raw_json_path:
        raise ValueError("data.json_path must be set")
    json_path = Path(str(raw_json_path))

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0 when set")

    samples: list[Dict[str, Any]] = []
    for record in _iter_tts_records(json_path):
        sample_id = str(record.get("id", ""))
        assistant_text = record.get("assistant_text")
        unit_ids = record.get("unit_ids")

        # Keep sample contract strict so eval failures are easy to trust.
        if not sample_id:
            raise ValueError("sample id is missing")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            raise ValueError(f"sample {sample_id}: assistant_text is missing or empty")
        if not isinstance(unit_ids, list) or not unit_ids:
            raise ValueError(f"sample {sample_id}: unit_ids must be a non-empty list")

        samples.append(
            {
                "id": sample_id,
                "assistant_text": assistant_text,
                "unit_ids": [int(unit) for unit in unit_ids],
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("no TTS samples found for evaluation")
    return samples


def _resolve_dtype(precision: str) -> torch.dtype | None:
    """Map training precision string to inference load dtype."""
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "no":
        return None
    raise ValueError("unsupported precision; use bf16 or no")


def _load_model_state(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from one Accelerate checkpoint directory."""
    bin_path = checkpoint_dir / "pytorch_model.bin"
    safe_path = checkpoint_dir / "model.safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    if safe_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load model.safetensors") from exc
        return load_file(str(safe_path))
    raise FileNotFoundError(f"no model checkpoint found in: {checkpoint_dir}")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop a leading 'module.' prefix when checkpoints were saved from wrapped modules."""
    if not state:
        return state
    if not all(key.startswith("module.") for key in state.keys()):
        return state
    return {key[len("module.") :]: value for key, value in state.items()}


def _build_model_and_tokenizer(cfg: Dict[str, Any]) -> tuple[torch.nn.Module, Any, TokenIds]:
    """Build TTS PomTTS model + tokenizer using the training config."""
    model_cfg = _get_cfg_section(cfg, "model")
    train_cfg = _get_cfg_section(cfg, "training")
    dtype = _resolve_dtype(str(train_cfg.get("precision", "bf16")))
    model, tokenizer, token_ids = build_pom_tts(
        base_model_id=str(model_cfg.get("id", "Qwen/Qwen3-0.6B")),
        base_cache_dir=model_cfg.get("cache"),
        dtype=dtype,
        speech_vocab_size=int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if model.speech_token_offset is None:
        raise ValueError("speech_token_offset is missing on speech_lm")
    return model, tokenizer, token_ids


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Load checkpoint weights into the TTS PomTTS model with a tiny fallback for key prefixes."""
    state = _load_model_state(checkpoint_dir)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Some wrapped runs save with a leading "module." key prefix.
        stripped = _strip_module_prefix(state)
        if stripped is state:
            raise
        model.load_state_dict(stripped, strict=True)


def _build_eval_sequence(
    sample: Dict[str, Any],
    *,
    tokenizer: Any,
    token_ids: TokenIds,
    speech_token_offset: int,
    read_length: int,
    write_length: int,
) -> tuple[list[int], list[int]]:
    """Build one TTS Read/Write sequence from one eval sample."""
    text_ids = tokenizer(str(sample["assistant_text"]), add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    return build_read_write_sequence(
        text_ids,
        sample["unit_ids"],
        speech_token_offset=int(speech_token_offset),
        sep_id=int(token_ids.sep_id),
        eos_id=int(tokenizer.eos_token_id),
        read_length=int(read_length),
        write_length=int(write_length),
    )


def _evaluate_one_sample(
    sample: Dict[str, Any],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    token_ids: TokenIds,
    speech_token_offset: int,
    speech_vocab_size: int,
    read_length: int,
    write_length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run teacher-forced TTS evaluation for one sample and return tiny metrics."""
    input_ids_list, labels_list = _build_eval_sequence(
        sample,
        tokenizer=tokenizer,
        token_ids=token_ids,
        speech_token_offset=speech_token_offset,
        read_length=read_length,
        write_length=write_length,
    )

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[0]  # [seq_len, vocab]
    next_token_pred = logits[:-1].argmax(dim=-1).tolist()

    target_token_ids: list[int] = []
    pred_token_ids: list[int] = []
    # Causal LM predicts position i from logits at i-1.
    for pos in range(1, len(labels_list)):
        target = int(labels_list[pos])
        if target == IGNORE_INDEX:
            continue
        target_token_ids.append(target)
        pred_token_ids.append(int(next_token_pred[pos - 1]))
    if not target_token_ids:
        raise ValueError(f"sample {sample['id']}: no supervised target tokens found")

    eos_id = int(tokenizer.eos_token_id)
    if target_token_ids[-1] != eos_id:
        raise ValueError(f"sample {sample['id']}: final supervised token is not EOS")
    if len(target_token_ids) != len(sample["unit_ids"]) + 1:
        raise ValueError(f"sample {sample['id']}: supervised target length is inconsistent")

    correct = sum(int(pred == target) for pred, target in zip(pred_token_ids, target_token_ids))
    token_accuracy = float(correct) / float(len(target_token_ids))
    first_mismatch = next(
        (idx for idx, (pred, target) in enumerate(zip(pred_token_ids, target_token_ids)) if pred != target),
        None,
    )

    target_unit_ids: list[int] = []
    pred_unit_ids: list[int] = []
    invalid_pred_unit_count = 0
    min_speech_id = int(speech_token_offset)
    max_speech_id = int(speech_token_offset) + int(speech_vocab_size) - 1

    # Compare only speech positions (exclude EOS).
    for pred_id, target_id in zip(pred_token_ids, target_token_ids):
        if target_id == eos_id:
            continue
        if target_id < min_speech_id or target_id > max_speech_id:
            raise ValueError(f"sample {sample['id']}: target speech token id out of range")
        target_unit_ids.append(target_id - int(speech_token_offset))
        if min_speech_id <= pred_id <= max_speech_id:
            pred_unit_ids.append(pred_id - int(speech_token_offset))
        else:
            pred_unit_ids.append(-1)
            invalid_pred_unit_count += 1
    # Confirm sequence->label->unit mapping preserves exact target order.
    if target_unit_ids != sample["unit_ids"]:
        raise ValueError(f"sample {sample['id']}: target unit order mismatch; check read/write alignment")

    return {
        "sample_id": sample["id"],
        "assistant_text": sample["assistant_text"],
        "target_unit_ids": target_unit_ids,
        "pred_unit_ids": pred_unit_ids,
        "prediction_units": "".join(f"<{unit}>" for unit in pred_unit_ids if unit >= 0),
        "target_units": "".join(f"<{unit}>" for unit in target_unit_ids),
        "token_accuracy": token_accuracy,
        "exact_match": bool(pred_token_ids == target_token_ids),
        "eos_correct": bool(pred_token_ids[-1] == eos_id),
        "first_mismatch_index": first_mismatch,
        "invalid_pred_unit_count": int(invalid_pred_unit_count),
    }


def _write_results_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    """Write per-sample evaluation rows to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_results(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute tiny aggregate metrics for quick overfit pass/fail checks."""
    total = len(results)
    exact = sum(int(row["exact_match"]) for row in results)
    eos = sum(int(row["eos_correct"]) for row in results)
    invalid_units = sum(int(row["invalid_pred_unit_count"]) for row in results)
    mean_acc = sum(float(row["token_accuracy"]) for row in results) / float(total)
    return {
        "num_samples": total,
        "exact_match_rate": float(exact) / float(total),
        "mean_token_accuracy": mean_acc,
        "eos_correct_rate": float(eos) / float(total),
        "total_invalid_pred_units": int(invalid_units),
    }


def main() -> None:
    """Resolve inputs and run minimal teacher-forced TTS overfit evaluation."""
    args = _parse_args()
    cfg = _load_config(args.config)
    checkpoint_dir = _resolve_checkpoint_dir(cfg, args.checkpoint)
    samples = _load_eval_samples(cfg, limit=args.limit)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTS overfit evaluation")

    model, tokenizer, token_ids = _build_model_and_tokenizer(cfg)
    _load_checkpoint_into_model(model, checkpoint_dir)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    train_cfg = _get_cfg_section(cfg, "training")
    read_length = int(train_cfg.get("read_length", 3))
    write_length = int(train_cfg.get("write_length", 10))
    if read_length <= 0 or write_length <= 0:
        raise ValueError("training.read_length and training.write_length must be > 0")

    # Evaluate each overfit sample independently with teacher forcing.
    results = [
        _evaluate_one_sample(
            sample,
            model=model,
            tokenizer=tokenizer,
            token_ids=token_ids,
            speech_token_offset=int(model.speech_token_offset),
            speech_vocab_size=int(model.speech_vocab_size),
            read_length=read_length,
            write_length=write_length,
            device=device,
        )
        for sample in samples
    ]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_results_jsonl(output_path, results)
    summary = _summarize_results(results)

    print(f"checkpoint={checkpoint_dir}")
    print(f"output_jsonl={output_path}")
    print(f"num_samples={summary['num_samples']}")
    print(f"exact_match_rate={summary['exact_match_rate']:.4f}")
    print(f"mean_token_accuracy={summary['mean_token_accuracy']:.4f}")
    print(f"eos_correct_rate={summary['eos_correct_rate']:.4f}")
    print(f"total_invalid_pred_units={summary['total_invalid_pred_units']}")


if __name__ == "__main__":
    main()
