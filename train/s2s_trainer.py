"""S2S (Speech-to-Speech, Stage-2) trainer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from transformers import get_cosine_schedule_with_warmup

from model.constants import IGNORE_INDEX, SPEECH_VOCAB_SIZE
from model.pom_thinker import build_thinker
from model.pom_talker import PomTalker, build_talker
from model.tokenizers import TokenIds, build_pom_tokenizer, ensure_tokenizer_contract
from train.s2s_data import S2sCollator, S2sDataset
from train.s2s_sequence_builder import build_s2s_read_write_batch
from train.training_profiler import StepProfiler


def _get_cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return one config section as a dict."""
    section = cfg.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return section


def _load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a top-level dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping at the top level")
    return data


def _apply_path_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI path overrides onto the loaded config."""
    model_cfg = cfg.setdefault("model", {})
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})

    if args.json_path:
        data_cfg["json_path"] = args.json_path
    if args.audio_root:
        data_cfg["audio_root"] = args.audio_root
    if args.model_cache:
        model_cfg["cache"] = args.model_cache
    if args.speech_encoder_cache:
        model_cfg["speech_encoder_cache"] = args.speech_encoder_cache
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for Stage-2 training."""
    parser = argparse.ArgumentParser(description="S2S trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--json-path", type=str, default=None, help="Override data.json_path")
    parser.add_argument("--audio-root", type=str, default=None, help="Override data.audio_root")
    parser.add_argument("--model-cache", type=str, default=None, help="Override model.cache")
    parser.add_argument(
        "--speech-encoder-cache",
        type=str,
        default=None,
        help="Override model.speech_encoder_cache",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override training.output_dir")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path or 'latest'",
    )
    parser.add_argument(
        "--allow-new-wandb-run-on-resume",
        action="store_true",
        help="Allow starting a new wandb run when resumed checkpoint has no wandb_run_id",
    )
    return parser.parse_args()


def _ensure_output_dir(path: str) -> Path:
    """Create output dir and fail early if writes are not allowed."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    # Write/delete one tiny file to catch permission errors early.
    test_path = out / f".write_test.{os.getpid()}"
    test_path.write_text("ok", encoding="utf-8")
    test_path.unlink()
    return out


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    """Return the deterministic checkpoint path for a step."""
    return output_dir / "checkpoints" / f"step-{step:08d}"


def _resolve_resume_dir(output_dir: Path, resume: Optional[str]) -> Optional[Path]:
    """Resolve resume input into one concrete checkpoint directory."""
    if not resume:
        return None
    if resume != "latest":
        path = Path(resume)
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path

    root = output_dir / "checkpoints"
    candidates = sorted(path for path in root.glob("step-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under: {root}")
    return candidates[-1]


def _load_trainer_state(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load trainer_state.json from a checkpoint directory."""
    path = checkpoint_dir / "trainer_state.json"
    if not path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in: {checkpoint_dir}")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"trainer_state.json must be a JSON object: {path}")
    return state


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so builder defaults remain active."""
    return {key: value for key, value in data.items() if value is not None}


def _ensure_pad_token(tokenizer: Any) -> None:
    """Require a real pad token id for consistent padding behavior."""
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")


def _load_checkpoint_model_state(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load model weights from one checkpoint directory (pytorch_model.bin only)."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")
    model_path = ckpt_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            f"pytorch_model.bin not found in checkpoint: {ckpt_dir}"
        )
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"model checkpoint must be a state_dict mapping: {model_path}")
    return state


def _strict_load_module_checkpoint(module: torch.nn.Module, checkpoint_dir: str, *, name: str) -> None:
    """Load one checkpoint into a module with strict key matching."""
    state = _load_checkpoint_model_state(checkpoint_dir)
    try:
        module.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"{name} checkpoint key mismatch from {checkpoint_dir}") from exc


def _freeze_thinker_parameters(thinker: torch.nn.Module) -> None:
    """Freeze all Thinker parameters so Stage-2 never updates them."""
    for param in thinker.parameters():
        param.requires_grad = False


def _set_frozen_thinker_eval(thinker: torch.nn.Module) -> None:
    """Force the full frozen Thinker path into eval mode to disable dropout."""
    thinker.eval()
    speech_encoder = thinker.get_speech_encoder()
    if speech_encoder is not None:
        speech_encoder.eval()
    speech_projector = thinker.get_speech_projector()
    if speech_projector is not None:
        speech_projector.eval()


def _validate_thinker_stage2_contract(thinker: torch.nn.Module) -> None:
    """Fail fast unless Thinker exposes the small interface Stage-2 needs."""
    # Stage-2 depends on speech expansion helpers from the Thinker wrapper.
    required_methods = (
        "prepare_inputs_labels_for_speech_and_text",
        "get_speech_encoder",
        "get_speech_projector",
    )
    missing = [name for name in required_methods if not callable(getattr(thinker, name, None))]
    if missing:
        raise TypeError("thinker is missing required Stage-2 methods: " + ", ".join(missing))

    # Stage-2 frozen forward uses the HF-native decoder backbone entrypoint.
    base_model = getattr(thinker, "base_model", None)
    if base_model is None or not callable(base_model):
        raise TypeError("thinker must expose a callable base_model for Stage-2 frozen forward")


def _build_s2s_modules_and_tokenizer(
    cfg: Dict[str, Any],
) -> tuple[torch.nn.Module, PomTalker, Any, TokenIds]:
    """Build frozen Thinker + trainable Stage-2 PomTalker with one shared tokenizer."""
    model_cfg = _get_cfg_section(cfg, "model")

    base_model_id = str(model_cfg.get("id", "Qwen/Qwen3-0.6B"))
    base_cache = model_cfg.get("cache")
    tokenizer = build_pom_tokenizer(
        base_model_id=base_model_id,
        cache_dir=base_cache,
    )
    token_ids = ensure_tokenizer_contract(tokenizer)
    _ensure_pad_token(tokenizer)

    # Build Thinker with speech modules attached from Stage-1 settings.
    speech_spec = _drop_none(
        {
            "encoder_id": model_cfg.get("speech_encoder_id"),
            "encoder_cache": model_cfg.get("speech_encoder_cache"),
            "frame_stack": model_cfg.get("frame_stack"),
            "projector_hidden_dim": model_cfg.get("adapter_hidden_dim"),
        }
    )
    thinker, _, _ = build_thinker(
        base_model_id=base_model_id,
        cache_dir=base_cache,
        speech=speech_spec,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    _validate_thinker_stage2_contract(thinker)
    thinker.config.use_cache = False

    # Build Stage-2 talker with shared tokenizer/token ids.
    talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size),
        base_model_id=base_model_id,
        base_cache_dir=base_cache,
        speech_vocab_size=int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    if not isinstance(talker, PomTalker):
        raise TypeError("build_talker must return PomTalker")
    # Disable KV cache during Stage-2 training for stable memory usage.
    talker.config.use_cache = False
    _freeze_thinker_parameters(thinker)
    _set_frozen_thinker_eval(thinker)
    return thinker, talker, tokenizer, token_ids


def _resolve_tts_checkpoint(train_cfg: Dict[str, Any]) -> str:
    """Require one explicit Stage-2 PomTTS checkpoint path from config."""
    tts_checkpoint = train_cfg.get("tts_checkpoint")
    # Keep Thinker/TTS bootstrap symmetric: one strict checkpoint path each.
    if tts_checkpoint is None:
        raise ValueError("training.tts_checkpoint must be set for Stage-2 runs")
    if not isinstance(tts_checkpoint, str):
        raise ValueError("training.tts_checkpoint must be a string path")
    return str(tts_checkpoint)


def _bootstrap_stage2_from_stage1_checkpoints(
    cfg: Dict[str, Any],
    *,
    thinker: torch.nn.Module,
    talker: PomTalker,
) -> None:
    """Bootstrap fresh Stage-2 run from SU Thinker and Stage-1b PomTTS checkpoints."""
    train_cfg = _get_cfg_section(cfg, "training")
    thinker_checkpoint = train_cfg.get("thinker_checkpoint")
    tts_checkpoint = _resolve_tts_checkpoint(train_cfg)
    if not thinker_checkpoint or not isinstance(thinker_checkpoint, str):
        raise ValueError("training.thinker_checkpoint must be set for fresh Stage-2 runs")

    # Keep Thinker bootstrap as strict full-module load.
    _strict_load_module_checkpoint(thinker, thinker_checkpoint, name="thinker")
    _strict_bootstrap_talker_from_tts_checkpoint(talker, tts_checkpoint)


def _strict_bootstrap_talker_from_tts_checkpoint(
    talker: PomTalker,
    tts_checkpoint_dir: str,
) -> None:
    """Load Stage-1b TTS weights into Talker with strict non-gate key matching."""
    state = _load_checkpoint_model_state(tts_checkpoint_dir)
    # Allow only gate-fusion params to be missing; all LM keys must match exactly.
    try:
        load_result = talker.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        # Normalize tensor-shape/type load failures to one consistent bootstrap contract.
        raise ValueError(
            f"tts checkpoint key mismatch from {tts_checkpoint_dir}: {exc}"
        ) from exc
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    expected_missing = {
        key for key in talker.state_dict().keys() if key.startswith("gate_fusion.")
    }
    if unexpected_keys:
        raise ValueError(
            f"tts checkpoint key mismatch from {tts_checkpoint_dir}: unexpected keys {sorted(unexpected_keys)}"
        )
    if missing_keys != expected_missing:
        raise ValueError(
            f"tts checkpoint key mismatch from {tts_checkpoint_dir}: "
            f"missing keys {sorted(missing_keys)} do not match expected gate-only keys {sorted(expected_missing)}"
        )


def _build_resume_compat(cfg: Dict[str, Any], *, grad_accum: int, precision: str) -> Dict[str, Any]:
    """Build minimal run-shape metadata for resume consistency checks."""
    model_cfg = _get_cfg_section(cfg, "model")
    data_cfg = _get_cfg_section(cfg, "data")
    train_cfg = _get_cfg_section(cfg, "training")
    thinker_checkpoint = train_cfg.get("thinker_checkpoint")
    tts_checkpoint = _resolve_tts_checkpoint(train_cfg)
    if not thinker_checkpoint:
        raise ValueError("training.thinker_checkpoint must be set")
    return {
        "model_id": str(model_cfg.get("id", "Qwen/Qwen3-0.6B")),
        "speech_encoder_id": str(model_cfg.get("speech_encoder_id", "openai/whisper-large-v3")),
        "frame_stack": model_cfg.get("frame_stack"),
        "adapter_hidden_dim": model_cfg.get("adapter_hidden_dim"),
        "speech_vocab_size": int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
        "data_json_path": str(Path(str(data_cfg.get("json_path", ""))).resolve()),
        "data_audio_root": str(Path(str(data_cfg.get("audio_root", ""))).resolve()),
        "shuffle_shards": bool(data_cfg.get("shuffle_shards", False)),
        "shuffle_seed": data_cfg.get("shuffle_seed"),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "grad_accum": int(grad_accum),
        "precision": str(precision),
        "read_length": int(train_cfg.get("read_length", 3)),
        "write_length": int(train_cfg.get("write_length", 10)),
        "training_steps": int(train_cfg.get("steps", 0)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "lr": float(train_cfg.get("lr", 1.0e-3)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        # Keep bootstrap sources explicit so accidental checkpoint mixups fail fast.
        "thinker_checkpoint": str(Path(str(thinker_checkpoint)).resolve()),
        "tts_checkpoint": str(Path(str(tts_checkpoint)).resolve()),
    }


def _validate_resume_compat(resume_state: Dict[str, Any], current_compat: Dict[str, Any]) -> None:
    """Fail fast if resume checkpoint metadata differs from current config."""
    resume_compat = resume_state.get("compat")
    if not isinstance(resume_compat, dict):
        raise ValueError("checkpoint is missing compatibility metadata in trainer_state.json")
    for key, current_value in current_compat.items():
        if resume_compat.get(key) != current_value:
            raise ValueError(
                f"resume mismatch for {key}: "
                f"checkpoint={resume_compat.get(key)!r} current={current_value!r}"
            )


def _build_dataloader(
    cfg: Dict[str, Any],
    *,
    tokenizer: Any,
    token_ids: TokenIds,
    read_length: int,
    write_length: int,
    start_shard_cursor: int = 0,
    shuffle_shards: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_epoch: int = 0,
) -> tuple[DataLoader, S2sDataset]:
    """Build Stage-2 dataset and dataloader."""
    data_cfg = _get_cfg_section(cfg, "data")
    train_cfg = _get_cfg_section(cfg, "training")

    json_path = data_cfg.get("json_path")
    audio_root = data_cfg.get("audio_root")
    if not json_path or not audio_root:
        raise ValueError("data.json_path and data.audio_root must be set")

    # Enable runtime ratio filtering in dataset to avoid collator hard-fail drops.
    dataset = S2sDataset(
        json_path=str(json_path),
        audio_root=str(audio_root),
        start_shard_cursor=int(start_shard_cursor),
        shuffle_shards=bool(shuffle_shards),
        shuffle_seed=shuffle_seed,
        shuffle_epoch=int(shuffle_epoch),
        tokenizer=tokenizer,
        read_length=int(read_length),
        write_length=int(write_length),
    )
    collator = S2sCollator(
        tokenizer,
        token_ids=token_ids,
        read_length=int(read_length),
        write_length=int(write_length),
        ignore_index=IGNORE_INDEX,
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    persistent_workers = bool(train_cfg.get("dataloader_persistent_workers", False))
    pin_memory = bool(train_cfg.get("dataloader_pin_memory", False))
    prefetch_factor = train_cfg.get("dataloader_prefetch_factor")

    # Keep dataloader knobs fully config-driven for hardware tuning.
    dataloader_kwargs: Dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "collate_fn": collator,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader, dataset


def _drop_non_model_fields(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Drop debugging/loader metadata not consumed by Stage-2 step logic."""
    model_batch = dict(batch)
    model_batch.pop("sample_ids", None)
    model_batch.pop("shard_indices", None)
    model_batch.pop("shard_cursors", None)
    model_batch.pop("worker_ids", None)
    return model_batch


def _validate_s2s_batch_contract(batch: Dict[str, Any]) -> None:
    """Validate one collated Stage-2 batch before training starts."""
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    content_ids = batch.get("content_ids")
    unit_ids = batch.get("unit_ids")
    waveforms = batch.get("speech_waveforms")
    sampling_rates = batch.get("speech_sampling_rate")
    if input_ids is None or labels is None or attention_mask is None:
        raise ValueError("batch must contain input_ids, labels, attention_mask")
    if content_ids is None or unit_ids is None:
        raise ValueError("batch must contain content_ids and unit_ids")
    if waveforms is None or sampling_rates is None:
        raise ValueError("batch must contain speech_waveforms and speech_sampling_rate")
    if input_ids.ndim != 2 or labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids, labels, attention_mask must be rank-2 tensors")
    if input_ids.shape != labels.shape or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids, labels, attention_mask must have the same shape")
    if int(input_ids.size(0)) <= 0:
        raise ValueError("batch is empty")
    if not isinstance(content_ids, list) or not isinstance(unit_ids, list):
        raise ValueError("content_ids and unit_ids must be lists")
    if len(content_ids) != int(input_ids.size(0)) or len(unit_ids) != int(input_ids.size(0)):
        raise ValueError("content_ids and unit_ids must match batch size")
    if not isinstance(waveforms, list) or not isinstance(sampling_rates, list):
        raise ValueError("speech_waveforms and speech_sampling_rate must be lists")
    if len(waveforms) != len(sampling_rates):
        raise ValueError("speech_waveforms and speech_sampling_rate length mismatch")
    if int((labels != IGNORE_INDEX).sum().item()) <= 0:
        raise ValueError("batch has no supervised tokens")


def _make_sample_id_logger(
    output_dir: Path, accelerator: Accelerator, *, enabled: bool
):
    """Build optional sample-id logger for sharding/debug checks."""
    if not enabled:
        return lambda _sample_ids: None

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    max_ids = 100
    seen = 0
    rank = accelerator.process_index
    path = debug_dir / f"sample_ids.rank{rank}.jsonl"

    def _log(sample_ids):
        """Append a small bounded sample-id trace."""
        nonlocal seen
        if not sample_ids or seen >= max_ids:
            return
        with path.open("a", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                if seen >= max_ids:
                    break
                handle.write(json.dumps({"sample_id": sample_id}) + "\n")
                seen += 1

    return _log


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    output_dir: Path,
    step: int,
    grad_accum: int,
    micro_batches_seen: int,
    wandb_run_id: Optional[str],
    next_shard_cursor: int,
    shuffle_epoch: int,
    compat: Dict[str, Any],
    keep_last_n: int,
) -> None:
    """Save accelerate state and compact trainer metadata for resume."""
    checkpoint_dir = _checkpoint_dir(output_dir, step)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir), safe_serialization=False)

    if accelerator.is_main_process:
        trainer_state = {
            "global_step": int(step),
            "grad_accum": int(grad_accum),
            "micro_batches_seen": int(micro_batches_seen),
            "wandb_run_id": wandb_run_id,
            "next_shard_cursor": int(next_shard_cursor),
            "shuffle_epoch": int(shuffle_epoch),
            "compat": compat,
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2),
            encoding="utf-8",
        )
        checkpoints = sorted(path for path in checkpoint_dir.parent.glob("step-*") if path.is_dir())
        keep_count = max(int(keep_last_n), 1)
        for stale in checkpoints[:-keep_count]:
            shutil.rmtree(stale)
    accelerator.wait_for_everyone()


def _expand_speech_inputs(
    *,
    thinker: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: torch.Tensor,
    speech_waveforms: Sequence[torch.Tensor],
    speech_sampling_rate: Optional[int | Sequence[int]],
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Expand <speech> sentinels into speech-frame embeddings via frozen Thinker."""
    (
        _,
        position_ids,
        expanded_attention,
        _,
        inputs_embeds,
        expanded_labels,
    ) = thinker.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        speech_waveforms=speech_waveforms,
        speech_sampling_rate=speech_sampling_rate,
    )
    if position_ids is None or inputs_embeds is None or expanded_labels is None:
        raise RuntimeError("thinker speech expansion returned missing tensors")
    return position_ids, expanded_attention, inputs_embeds, expanded_labels


def _run_frozen_thinker(
    *,
    thinker: torch.nn.Module,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    """Run frozen Thinker backbone and return final hidden states."""
    # Use the HF-native decoder backbone instead of legacy get_model() wrappers.
    thinker_backbone = getattr(thinker, "base_model", None)
    if thinker_backbone is None or not callable(thinker_backbone):
        raise TypeError("thinker must expose a callable base_model for Stage-2 frozen forward")
    outputs = thinker_backbone(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        return_dict=True,
    )
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise RuntimeError("thinker backbone forward did not return last_hidden_state")
    return last_hidden_state


def _find_unique_subsequence_start(*, sequence: torch.Tensor, target: torch.Tensor) -> int:
    """Return the unique start index of target in sequence, or fail on none/ambiguous."""
    target_len = int(target.numel())
    if target_len <= 0:
        raise ValueError("target must be non-empty")
    if int(sequence.numel()) < target_len:
        raise ValueError("no match")

    # Vectorized contiguous matching keeps comparison on-device for GPU efficiency.
    windows = sequence.unfold(0, target_len, 1)
    match_mask = windows.eq(target).all(dim=1)
    match_indices = match_mask.nonzero(as_tuple=False).flatten()
    num_matches = int(match_indices.numel())
    if num_matches == 0:
        raise ValueError("no match")
    if num_matches > 1:
        raise ValueError(f"ambiguous ({num_matches} matches)")
    return int(match_indices[0].item())


def _extract_aligned_reply_states(
    *,
    last_hidden: torch.Tensor,
    expanded_labels: torch.Tensor,
    content_ids_rows: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Select reply hidden states and align them to plain assistant content ids."""
    aligned: list[torch.Tensor] = []
    for row_idx, content_ids in enumerate(content_ids_rows):
        # Non-IGNORE labels mark reply-token positions after speech expansion.
        reply_mask = expanded_labels[row_idx] != IGNORE_INDEX
        reply_ids = expanded_labels[row_idx][reply_mask]
        reply_states = last_hidden[row_idx][reply_mask]

        content_ids = torch.as_tensor(content_ids, dtype=torch.long, device=reply_ids.device)
        content_len = int(content_ids.numel())
        if content_len <= 0:
            raise ValueError(f"content_ids[{row_idx}] must be non-empty")
        if int(reply_ids.numel()) < content_len:
            raise ValueError(
                f"sample {row_idx}: reply token count {int(reply_ids.numel())} < content token count {content_len}"
            )

        # Align to the unique content span inside template reply ids.
        try:
            content_start = _find_unique_subsequence_start(sequence=reply_ids, target=content_ids)
        except ValueError as exc:
            reason = str(exc)
            raise ValueError(
                f"sample {row_idx}: alignment failed: {reason} "
                f"(reply_len={int(reply_ids.numel())}, content_len={content_len})"
            ) from exc
        aligned.append(reply_states[content_start : content_start + content_len])
    return aligned


def _run_stage2_step(
    *,
    thinker: torch.nn.Module,
    talker: PomTalker,
    batch: Dict[str, Any],
    sep_id: int,
    eos_id: int,
    read_length: int,
    write_length: int,
) -> tuple[Any, torch.Tensor]:
    """Run one explicit Stage-2 step: frozen Thinker -> fusion -> Read/Write -> Talker LM."""
    content_ids = batch["content_ids"]
    unit_ids = batch["unit_ids"]
    _set_frozen_thinker_eval(thinker)

    # Run frozen Thinker path without gradient tracking.
    with torch.no_grad():
        position_ids, expanded_attention, inputs_embeds, expanded_labels = _expand_speech_inputs(
            thinker=thinker,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            speech_waveforms=batch["speech_waveforms"],
            speech_sampling_rate=batch["speech_sampling_rate"],
        )
        last_hidden = _run_frozen_thinker(
            thinker=thinker,
            position_ids=position_ids,
            attention_mask=expanded_attention,
            inputs_embeds=inputs_embeds,
        )
        hidden_rows = _extract_aligned_reply_states(
            last_hidden=last_hidden,
            expanded_labels=expanded_labels,
            content_ids_rows=content_ids,
        )

    # Fuse Thinker hidden states with Talker text embeddings (Stage-2 module role).
    fused_rows = talker.fuse(hidden_rows=hidden_rows, content_ids=content_ids)
    talker_batch = build_s2s_read_write_batch(
        talker=talker,
        fused_rows=fused_rows,
        unit_rows=unit_ids,
        sep_id=int(sep_id),
        eos_id=int(eos_id),
        read_length=int(read_length),
        write_length=int(write_length),
        ignore_index=IGNORE_INDEX,
    )
    # Run only Talker LM generation here; all non-generation logic stays trainer-owned.
    outputs = talker(**talker_batch, return_dict=True, use_cache=False)

    # Count target tokens from talker labels for token-weighted loss logging.
    num_target_tokens = (talker_batch["labels"] != IGNORE_INDEX).sum()
    if int(num_target_tokens.item()) <= 0:
        raise ValueError("talker labels have no supervised target tokens")
    return outputs, num_target_tokens


def main() -> None:
    """Run Stage-2 (S2S) training."""
    args = _parse_args()
    cfg = _load_config(args.config)
    _apply_path_overrides(cfg, args)
    train_cfg = _get_cfg_section(cfg, "training")
    data_cfg = _get_cfg_section(cfg, "data")

    output_dir = train_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("training.output_dir must be set")
    output_dir = _ensure_output_dir(str(output_dir))
    resume_dir = _resolve_resume_dir(output_dir, args.resume)

    # Restore trainer progress metadata when resuming.
    resume_state: Optional[Dict[str, Any]] = None
    resume_step = 0
    micro_batches_seen = 0
    resume_next_shard_cursor = 0
    resume_shuffle_epoch = 0
    resume_wandb_run_id: Optional[str] = None
    if resume_dir is not None:
        resume_state = _load_trainer_state(resume_dir)
        resume_step = int(resume_state.get("global_step", 0))
        micro_batches_seen = int(resume_state.get("micro_batches_seen", 0))
        resume_next_shard_cursor = int(resume_state.get("next_shard_cursor", 0))
        resume_shuffle_epoch = int(resume_state.get("shuffle_epoch", 0))
        resume_wandb_run_id = resume_state.get("wandb_run_id")

    # Resolve shard shuffle settings before building compatibility metadata.
    shuffle_shards = bool(data_cfg.get("shuffle_shards", False))
    shuffle_seed = data_cfg.get("shuffle_seed")
    if shuffle_shards and shuffle_seed is None:
        shuffle_seed = train_cfg.get("seed")
    if shuffle_shards and shuffle_seed is None:
        raise ValueError("data.shuffle_seed (or training.seed) must be set when shuffle_shards is enabled")
    if shuffle_seed is not None:
        data_cfg["shuffle_seed"] = int(shuffle_seed)
    data_cfg["shuffle_shards"] = shuffle_shards

    grad_accum = int(train_cfg.get("grad_accum", 1))
    precision = str(train_cfg.get("precision", "bf16")).lower()
    if precision not in {"bf16", "no"}:
        raise ValueError("unsupported precision; use bf16 or no")
    read_length = int(train_cfg.get("read_length", 3))
    write_length = int(train_cfg.get("write_length", 10))
    if read_length <= 0 or write_length <= 0:
        raise ValueError("training.read_length and training.write_length must be > 0")

    resume_compat = _build_resume_compat(cfg, grad_accum=grad_accum, precision=precision)
    if resume_state is not None:
        _validate_resume_compat(resume_state, resume_compat)

    wandb_cfg = _get_cfg_section(cfg, "wandb")
    log_with = "wandb" if wandb_cfg.get("enabled", False) else None
    dataloader_non_blocking = bool(train_cfg.get("dataloader_non_blocking", False))
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=False,
        non_blocking=dataloader_non_blocking,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=precision,
        dataloader_config=dataloader_config,
        log_with=log_with,
    )
    debug_sample_ids = bool(train_cfg.get("debug_sample_ids", False))
    log_sample_ids = _make_sample_id_logger(output_dir, accelerator, enabled=debug_sample_ids)

    # Set global random seed for DDP-consistent runs.
    seed = train_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Build Stage-2 modules and bootstrap only on fresh runs.
    thinker, talker, tokenizer, token_ids = _build_s2s_modules_and_tokenizer(cfg)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if resume_dir is None:
        _bootstrap_stage2_from_stage1_checkpoints(cfg, thinker=thinker, talker=talker)

    json_path = Path(str(data_cfg.get("json_path", "")))
    data_is_sharded = json_path.is_dir()
    if shuffle_shards and not data_is_sharded:
        raise ValueError("data.shuffle_shards requires json_path to be a shard directory")
    if resume_next_shard_cursor < 0:
        raise ValueError(
            f"checkpoint next_shard_cursor must be >= 0, got {resume_next_shard_cursor}"
        )
    if data_is_sharded:
        shard_count = len(sorted(json_path.glob("shard-*.jsonl")))
        if shard_count <= 0:
            raise FileNotFoundError(f"no shard-*.jsonl files found in: {json_path}")
        if resume_next_shard_cursor >= shard_count:
            if accelerator.is_main_process:
                accelerator.print(
                    "resume shard cursor reached end of shard list; wrapping to shard 0 "
                    f"(saved={resume_next_shard_cursor}, num_shards={shard_count})"
                )
            resume_next_shard_cursor = 0
            if shuffle_shards:
                resume_shuffle_epoch += 1
    else:
        resume_next_shard_cursor = 0

    dataloader, dataset = _build_dataloader(
        cfg,
        tokenizer=tokenizer,
        token_ids=token_ids,
        read_length=read_length,
        write_length=write_length,
        start_shard_cursor=resume_next_shard_cursor,
        shuffle_shards=shuffle_shards,
        shuffle_seed=shuffle_seed,
        shuffle_epoch=resume_shuffle_epoch,
    )

    lr = float(train_cfg.get("lr", 1.0e-3))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    if max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be > 0")
    trainable_params = [param for param in talker.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = train_cfg.get("steps")
    if total_steps is None:
        raise ValueError("training.steps must be set for S2S training")
    total_steps = int(total_steps)
    if resume_step > total_steps:
        raise ValueError(
            f"checkpoint step {resume_step} is greater than training.steps {total_steps}"
        )
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    thinker, talker, optimizer, dataloader = accelerator.prepare(thinker, talker, optimizer, dataloader)
    optim_params = [param for group in optimizer.param_groups for param in group["params"]]
    accelerator.register_for_checkpointing(scheduler)
    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))

    talker.train()
    _set_frozen_thinker_eval(thinker)

    wandb_run_id = resume_wandb_run_id
    if wandb_cfg.get("enabled", False):
        if resume_dir is not None and not wandb_run_id:
            if not args.allow_new_wandb_run_on_resume:
                raise ValueError(
                    "resume checkpoint is missing wandb_run_id; "
                    "pass --allow-new-wandb-run-on-resume to override"
                )
            if accelerator.is_main_process:
                accelerator.print(
                    "resume checkpoint has no wandb_run_id; starting a new wandb run due to "
                    "--allow-new-wandb-run-on-resume"
                )
        if not wandb_run_id:
            wandb_run_id = uuid.uuid4().hex
    if wandb_cfg.get("enabled", False):
        wandb_init: Dict[str, Any] = {
            "name": wandb_cfg.get("run_name"),
            "id": wandb_run_id,
        }
        if resume_dir is not None and resume_wandb_run_id:
            wandb_init["resume"] = "must"
        accelerator.init_trackers(
            project_name=str(wandb_cfg.get("project", "pom")),
            config=cfg,
            init_kwargs={"wandb": wandb_init},
        )

    save_every = train_cfg.get("save_every")
    save_every = int(save_every) if save_every is not None else None
    keep_last_n = int(train_cfg.get("keep_last_n_checkpoints", 5))
    if keep_last_n < 0:
        raise ValueError("training.keep_last_n_checkpoints must be >= 0")
    profile_enabled = bool(train_cfg.get("profile", True))

    # Fail fast before long runs if one batch contract/loss is broken.
    batch_iter = iter(dataloader)
    batch = next(batch_iter, None)
    if batch is None:
        raise ValueError("dataset produced no samples")
    _validate_s2s_batch_contract(batch)
    log_sample_ids(batch.get("sample_ids"))
    sanity_batch = _drop_non_model_fields(batch)
    with torch.no_grad():
        with accelerator.autocast():
            outputs, sanity_target_tokens = _run_stage2_step(
                thinker=thinker,
                talker=talker,
                batch=sanity_batch,
                sep_id=int(token_ids.sep_id),
                eos_id=int(eos_id),
                read_length=read_length,
                write_length=write_length,
            )
    loss = outputs.loss
    if loss is None or torch.isnan(loss).any():
        raise ValueError("loss is NaN or missing on the first batch")
    if loss.item() <= 0:
        raise ValueError("loss is zero on the first batch; check labels")
    if int(sanity_target_tokens.item()) <= 0:
        raise ValueError("sanity batch has no supervised talker tokens")
    if accelerator.is_main_process:
        accelerator.print(f"sanity loss: {loss.item():.4f}")

    # Run one backward pass check to ensure thinker remains fully frozen.
    optimizer.zero_grad(set_to_none=True)
    with accelerator.autocast():
        outputs, _ = _run_stage2_step(
            thinker=thinker,
            talker=talker,
            batch=sanity_batch,
            sep_id=int(token_ids.sep_id),
            eos_id=int(eos_id),
            read_length=read_length,
            write_length=write_length,
        )
    freeze_loss = outputs.loss
    if freeze_loss is None:
        raise ValueError("loss is missing during frozen-path sanity backward")
    accelerator.backward(freeze_loss)
    unwrapped_thinker = accelerator.unwrap_model(thinker)
    thinker_has_grad = any(param.grad is not None for param in unwrapped_thinker.parameters())
    if thinker_has_grad:
        raise ValueError("frozen thinker accumulated gradients during Stage-2 sanity backward")
    optimizer.zero_grad(set_to_none=True)

    # Main fixed-step training loop.
    step = resume_step
    next_shard_cursor = resume_next_shard_cursor if data_is_sharded else 0
    shuffle_epoch = resume_shuffle_epoch
    num_workers = int(train_cfg.get("num_workers", 0))
    worker_count = max(num_workers, 1)
    last_cursor_seen = {
        worker_id: int(next_shard_cursor) for worker_id in range(worker_count)
    }
    data_iter = iter(dataloader)
    use_cuda = accelerator.device.type == "cuda"
    effective_target = int(train_cfg.get("batch_size", 1)) * accelerator.num_processes * grad_accum
    stats = StepProfiler(
        enabled=profile_enabled,
        use_cuda=use_cuda,
        device=accelerator.device,
        world_size=accelerator.num_processes,
        grad_accum=accelerator.gradient_accumulation_steps,
        effective_target=effective_target,
    )
    loss_sum_local = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    target_tokens_local = torch.zeros((), device=accelerator.device, dtype=torch.int64)
    while step < total_steps:
        stats.start_window()
        try:
            if profile_enabled:
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_time = time.perf_counter() - data_wait_start
            else:
                batch = next(data_iter)
        except StopIteration:
            # Reset iterator and sharded cursor when dataset is exhausted.
            if data_is_sharded:
                next_shard_cursor = 0
                dataset.start_shard_cursor = 0
                last_cursor_seen = {
                    worker_id: int(next_shard_cursor) for worker_id in range(worker_count)
                }
                if shuffle_shards:
                    shuffle_epoch += 1
                    dataset.shuffle_epoch = shuffle_epoch
            data_iter = iter(dataloader)
            if profile_enabled:
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_time = time.perf_counter() - data_wait_start
            else:
                batch = next(data_iter)

        micro_batches_seen += 1
        shard_cursors = batch.get("shard_cursors")
        worker_ids = batch.get("worker_ids")
        if data_is_sharded and shard_cursors and worker_ids:
            # Track conservative shard progress across workers and ranks.
            for cursor, worker_id in zip(shard_cursors, worker_ids):
                wid = int(worker_id)
                cur = int(cursor)
                prev = last_cursor_seen.get(wid, next_shard_cursor)
                if cur > prev:
                    last_cursor_seen[wid] = cur
            local_safe_cursor = min(last_cursor_seen.values())
            shard_tensor = torch.tensor(float(local_safe_cursor), device=accelerator.device)
            global_safe_cursor = accelerator.reduce(shard_tensor, reduction="min")
            next_shard_cursor = int(global_safe_cursor.item())

        if profile_enabled:
            stats.record_batch(batch, data_time)

        with accelerator.accumulate(talker):
            log_sample_ids(batch.get("sample_ids"))
            model_batch = _drop_non_model_fields(batch)
            with accelerator.autocast():
                outputs, target_tokens_micro = _run_stage2_step(
                    thinker=thinker,
                    talker=talker,
                    batch=model_batch,
                    sep_id=int(token_ids.sep_id),
                    eos_id=int(eos_id),
                    read_length=read_length,
                    write_length=write_length,
                )
            loss = outputs.loss
            if loss is None:
                raise ValueError("loss is missing during training")

            # Weight logging by Talker-supervised tokens (not Thinker teacher-forcing labels).
            if not torch.is_tensor(target_tokens_micro):
                target_tokens_micro = torch.as_tensor(
                    target_tokens_micro,
                    device=accelerator.device,
                    dtype=torch.int64,
                )
            target_tokens_micro = target_tokens_micro.to(device=accelerator.device, dtype=torch.int64)
            if target_tokens_micro.ndim != 0:
                raise ValueError("num_target_tokens must be a scalar tensor")
            if int(target_tokens_micro.item()) <= 0:
                raise ValueError("num_target_tokens must be > 0")

            loss_sum_local += loss.detach().float() * target_tokens_micro.to(dtype=torch.float32)
            target_tokens_local += target_tokens_micro

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(optim_params, max_grad_norm)
                grad_norm_value = (
                    float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                global_loss_sum = accelerator.reduce(loss_sum_local, reduction="sum")
                global_tokens = accelerator.reduce(target_tokens_local, reduction="sum")
                num_target_tokens = int(global_tokens.item())
                if num_target_tokens <= 0:
                    raise ValueError("global target token count is zero; cannot compute train/loss")
                train_loss = global_loss_sum.item() / float(num_target_tokens)
                logs = {
                    "train/loss": train_loss,
                    "train/num_talker_target_tokens": num_target_tokens,
                    "train/num_target_tokens": num_target_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm_value,
                }
                logs.update(stats.finalize_step(accelerator))
                accelerator.log(logs, step=step)
                loss_sum_local.zero_()
                target_tokens_local.zero_()

                if save_every and step % save_every == 0:
                    _save_checkpoint(
                        accelerator=accelerator,
                        output_dir=output_dir,
                        step=step,
                        grad_accum=grad_accum,
                        micro_batches_seen=micro_batches_seen,
                        wandb_run_id=wandb_run_id,
                        next_shard_cursor=next_shard_cursor,
                        shuffle_epoch=shuffle_epoch,
                        compat=resume_compat,
                        keep_last_n=keep_last_n,
                    )

                if step >= total_steps:
                    break

    # Save terminal checkpoint for training resume/state handoff.
    _save_checkpoint(
        accelerator=accelerator,
        output_dir=output_dir,
        step=step,
        grad_accum=grad_accum,
        micro_batches_seen=micro_batches_seen,
        wandb_run_id=wandb_run_id,
        next_shard_cursor=next_shard_cursor,
        shuffle_epoch=shuffle_epoch,
        compat=resume_compat,
        keep_last_n=keep_last_n,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
