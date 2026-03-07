"""TTS (Text-to-Speech) trainer for PomTTS pretraining."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from model.constants import IGNORE_INDEX, SPEECH_VOCAB_SIZE
from model.pom_tts import build_pom_tts
from model.tokenizers import TokenIds
from train.tts_data import TtsCollator, TtsDataset
from train.training_profiler import StepProfiler


def _get_cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Read a nested config section or return an empty one."""
    section = cfg.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return section


def _load_config(path: str) -> Dict[str, Any]:
    """Load a simple YAML config into a dict."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping at the top level")
    return data


def _apply_path_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Override config paths with CLI flags when provided."""
    model_cfg = cfg.setdefault("model", {})
    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})

    if args.json_path:
        data_cfg["json_path"] = args.json_path
    if args.model_cache:
        model_cfg["cache"] = args.model_cache
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for TTS training."""
    parser = argparse.ArgumentParser(description="TTS trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--json-path", type=str, default=None, help="Override data.json_path")
    parser.add_argument("--model-cache", type=str, default=None, help="Override model.cache")
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
    """Create output dir if needed and assert it is writable."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    # Write/delete a tiny file so permission errors happen early.
    test_path = out / f".write_test.{os.getpid()}"
    test_path.write_text("ok", encoding="utf-8")
    test_path.unlink()
    return out


def _checkpoint_dir(output_dir: Path, step: int) -> Path:
    """Return deterministic per-step checkpoint path."""
    return output_dir / "checkpoints" / f"step-{step:08d}"


def _resolve_resume_dir(output_dir: Path, resume: Optional[str]) -> Optional[Path]:
    """Resolve resume input to one checkpoint directory."""
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
    """Load trainer_state.json from a checkpoint."""
    path = checkpoint_dir / "trainer_state.json"
    if not path.exists():
        raise FileNotFoundError(f"trainer_state.json not found in: {checkpoint_dir}")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"trainer_state.json must be a JSON object: {path}")
    return state


def _build_resume_compat(
    cfg: Dict[str, Any],
    *,
    grad_accum: int,
    precision: str,
    read_length: int,
    write_length: int,
) -> Dict[str, Any]:
    """Build compact config fingerprint for safe resume."""
    model_cfg = _get_cfg_section(cfg, "model")
    data_cfg = _get_cfg_section(cfg, "data")
    train_cfg = _get_cfg_section(cfg, "training")
    return {
        "model_id": str(model_cfg.get("id", "Qwen/Qwen3-0.6B")),
        "speech_vocab_size": int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
        "data_json_path": str(Path(str(data_cfg.get("json_path", ""))).resolve()),
        "shuffle_shards": bool(data_cfg.get("shuffle_shards", False)),
        "shuffle_seed": data_cfg.get("shuffle_seed"),
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "grad_accum": int(grad_accum),
        "precision": str(precision),
        "read_length": int(read_length),
        "write_length": int(write_length),
        # Keep scheduler-shaping fields fixed across resume.
        "training_steps": int(train_cfg.get("steps", 0)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "lr": float(train_cfg.get("lr", 5.0e-4)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
    }


def _validate_resume_compat(resume_state: Dict[str, Any], current_compat: Dict[str, Any]) -> None:
    """Fail fast if checkpoint and current config differ."""
    resume_compat = resume_state.get("compat")
    if not isinstance(resume_compat, dict):
        raise ValueError("checkpoint is missing compatibility metadata in trainer_state.json")
    for key, current_value in current_compat.items():
        if resume_compat.get(key) != current_value:
            raise ValueError(
                f"resume mismatch for {key}: "
                f"checkpoint={resume_compat.get(key)!r} current={current_value!r}"
            )


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
    """Save accelerate state plus minimal trainer metadata."""
    checkpoint_dir = _checkpoint_dir(output_dir, step)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir), safe_serialization=False)

    if accelerator.is_main_process:
        # Keep trainer state explicit so resume/debug is simple.
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
        # Retain only recent checkpoints; 0 means keep current only.
        checkpoints = sorted(path for path in checkpoint_dir.parent.glob("step-*") if path.is_dir())
        keep_count = max(int(keep_last_n), 1)
        for stale in checkpoints[:-keep_count]:
            shutil.rmtree(stale)
    accelerator.wait_for_everyone()


def _ensure_pad_token(tokenizer: Any) -> None:
    """Require a real pad token id for padding semantics."""
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")


def _build_model_and_tokenizer(cfg: Dict[str, Any]) -> tuple[torch.nn.Module, Any, TokenIds]:
    """Load PomTTS + tokenizer with TTS token contracts."""
    model_cfg = _get_cfg_section(cfg, "model")
    tts, tokenizer, token_ids = build_pom_tts(
        base_model_id=str(model_cfg.get("id", "Qwen/Qwen3-0.6B")),
        base_cache_dir=model_cfg.get("cache"),
        speech_vocab_size=int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
    )
    _ensure_pad_token(tokenizer)
    # Disable KV cache during training to avoid unnecessary memory growth.
    tts.config.use_cache = False
    return tts, tokenizer, token_ids


def _build_dataloader(
    cfg: Dict[str, Any],
    *,
    tokenizer: Any,
    token_ids: TokenIds,
    speech_token_offset: int,
    read_length: int,
    write_length: int,
    start_shard_cursor: int = 0,
    shuffle_shards: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_epoch: int = 0,
) -> tuple[DataLoader, TtsDataset]:
    """Construct the TTS dataset and dataloader."""
    data_cfg = _get_cfg_section(cfg, "data")
    train_cfg = _get_cfg_section(cfg, "training")
    json_path = data_cfg.get("json_path")
    if not json_path:
        raise ValueError("data.json_path must be set")

    max_seq_len = data_cfg.get("max_seq_len")
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)

    dataset = TtsDataset(
        json_path=str(json_path),
        start_shard_cursor=int(start_shard_cursor),
        shuffle_shards=bool(shuffle_shards),
        shuffle_seed=shuffle_seed,
        shuffle_epoch=int(shuffle_epoch),
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        read_length=int(read_length),
        write_length=int(write_length),
    )

    # Keep <sep> contract startup-resolved to match SU behavior.
    collator = TtsCollator(
        tokenizer,
        token_ids=token_ids,
        speech_token_offset=int(speech_token_offset),
        read_length=int(read_length),
        write_length=int(write_length),
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    persistent_workers = bool(train_cfg.get("dataloader_persistent_workers", False))
    pin_memory = bool(train_cfg.get("dataloader_pin_memory", False))
    prefetch_factor = train_cfg.get("dataloader_prefetch_factor")

    # Keep DataLoader knobs fully YAML-driven so hardware tuning is reproducible.
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


def _validate_tts_batch_contract(batch: Dict[str, Any], vocab_size: int) -> None:
    """Validate one collated batch contract before training."""
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if input_ids is None or labels is None:
        raise ValueError("batch must contain input_ids and labels")
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must be rank-2 tensors")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape")
    if input_ids.numel() == 0:
        raise ValueError("batch is empty")
    # Ensure model token lookups always stay in vocabulary bounds.
    if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= int(vocab_size):
        raise ValueError("input_ids contain out-of-range token ids")
    # Labels can be IGNORE_INDEX or valid token ids only.
    valid_labels = labels[labels != IGNORE_INDEX]
    if valid_labels.numel() == 0:
        raise ValueError("batch has no supervised target tokens")
    if int(valid_labels.min().item()) < 0 or int(valid_labels.max().item()) >= int(vocab_size):
        raise ValueError("labels contain out-of-range token ids")


def _drop_non_model_fields(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Drop debugging fields that model.forward does not consume."""
    model_batch = dict(batch)
    model_batch.pop("sample_ids", None)
    model_batch.pop("shard_indices", None)
    model_batch.pop("shard_cursors", None)
    model_batch.pop("worker_ids", None)
    return model_batch


def _make_sample_id_logger(
    output_dir: Path, accelerator: Accelerator, *, enabled: bool
):
    """Return a no-op logger unless debug sample logging is enabled."""
    if not enabled:
        return lambda _sample_ids: None

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    max_ids = 100
    seen = 0
    rank = accelerator.process_index
    path = debug_dir / f"sample_ids.rank{rank}.jsonl"

    def _log(sample_ids):
        """Append a few sample ids for sharding validation."""
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


def main() -> None:
    """Run TTS training."""
    args = _parse_args()
    cfg = _load_config(args.config)
    _apply_path_overrides(cfg, args)
    train_cfg = _get_cfg_section(cfg, "training")
    data_cfg = _get_cfg_section(cfg, "data")
    profile_enabled = bool(train_cfg.get("profile", True))
    output_dir = train_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("training.output_dir must be set")
    output_dir = _ensure_output_dir(str(output_dir))
    resume_dir = _resolve_resume_dir(output_dir, args.resume)

    # Start from zero unless resume metadata says otherwise.
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

    # Resolve shuffle settings early so resume compatibility is stable.
    shuffle_shards = bool(data_cfg.get("shuffle_shards", False))
    shuffle_seed = data_cfg.get("shuffle_seed")
    if shuffle_shards and shuffle_seed is None:
        shuffle_seed = train_cfg.get("seed")
    if shuffle_shards and shuffle_seed is None:
        raise ValueError("data.shuffle_seed (or training.seed) must be set when shuffle_shards is enabled")
    if shuffle_seed is not None:
        data_cfg["shuffle_seed"] = int(shuffle_seed)
    data_cfg["shuffle_shards"] = shuffle_shards

    # Accelerator owns device placement, precision, and DDP wiring.
    grad_accum = int(train_cfg.get("grad_accum", 1))
    precision = str(train_cfg.get("precision", "bf16")).lower()
    if precision not in {"bf16", "no"}:
        raise ValueError("unsupported precision; use bf16 or no")
    read_length = int(train_cfg.get("read_length", 3))
    write_length = int(train_cfg.get("write_length", 10))
    if read_length <= 0 or write_length <= 0:
        raise ValueError("training.read_length and training.write_length must be > 0")
    resume_compat = _build_resume_compat(
        cfg,
        grad_accum=grad_accum,
        precision=precision,
        read_length=read_length,
        write_length=write_length,
    )
    if resume_state is not None:
        _validate_resume_compat(resume_state, resume_compat)
    wandb_cfg = _get_cfg_section(cfg, "wandb")
    log_with = "wandb" if wandb_cfg.get("enabled", False) else None
    # Keep host->device copy behavior configurable from YAML.
    dataloader_non_blocking = bool(train_cfg.get("dataloader_non_blocking", False))
    # Control IterableDataset sharding and dataloader transfer behavior.
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
    log_sample_ids = _make_sample_id_logger(
        output_dir, accelerator, enabled=debug_sample_ids
    )

    # Set a global seed to improve reproducibility.
    seed = train_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Build the model/tokenizer and data pipeline.
    # Keep the resolved token contract from startup for downstream checks.
    model, tokenizer, token_ids = _build_model_and_tokenizer(cfg)
    if model.speech_token_offset is None:
        raise ValueError("speech_token_offset is missing on PomTTS")
    json_path = Path(str(data_cfg.get("json_path", "")))
    data_is_sharded = json_path.is_dir()
    if shuffle_shards and not data_is_sharded:
        raise ValueError("data.shuffle_shards requires json_path to be a shard directory")
    if resume_next_shard_cursor < 0:
        raise ValueError(
            f"checkpoint next_shard_cursor must be >= 0, got {resume_next_shard_cursor}"
        )
    if data_is_sharded:
        # Only shard directories participate in shard-cursor resume.
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
        # Single-file datasets always resume from shard 0.
        resume_next_shard_cursor = 0
    dataloader, dataset = _build_dataloader(
        cfg,
        tokenizer=tokenizer,
        token_ids=token_ids,
        speech_token_offset=int(model.speech_token_offset),
        read_length=read_length,
        write_length=write_length,
        start_shard_cursor=resume_next_shard_cursor,
        shuffle_shards=shuffle_shards,
        shuffle_seed=shuffle_seed,
        shuffle_epoch=resume_shuffle_epoch,
    )

    # Optimizer + scheduler are step-based.
    lr = float(train_cfg.get("lr", 5.0e-4))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    if max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be > 0")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total_steps = train_cfg.get("steps")
    if total_steps is None:
        raise ValueError("training.steps must be set for TTS training")
    total_steps = int(total_steps)
    if resume_step > total_steps:
        raise ValueError(
            f"checkpoint step {resume_step} is greater than training.steps {total_steps}"
        )
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Prepare model/optimizer/dataloader for distributed training.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    optim_params = [param for group in optimizer.param_groups for param in group["params"]]
    accelerator.register_for_checkpointing(scheduler)
    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))

    # Put model in training mode.
    model.train()

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

    # Init wandb tracker only when requested.
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

    # Fail-fast: ensure one batch works and loss is valid.
    batch_iter = iter(dataloader)
    batch = next(batch_iter, None)
    if batch is None:
        raise ValueError("dataset produced no samples")
    _validate_tts_batch_contract(batch, vocab_size=int(model.total_vocab_size))
    labels = batch["labels"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    # TTS training must supervise EOS stop prediction.
    if not torch.any(labels == int(eos_id)):
        raise ValueError("first batch has no supervised EOS token")
    log_sample_ids(batch.get("sample_ids"))
    sanity_batch = _drop_non_model_fields(batch)
    with torch.no_grad():
        with accelerator.autocast():
            outputs = model(**sanity_batch)
    loss = outputs.loss
    if loss is None or torch.isnan(loss).any():
        raise ValueError("loss is NaN or missing on the first batch")
    if loss.item() <= 0:
        raise ValueError("loss is zero on the first batch; check labels")
    if accelerator.is_main_process:
        accelerator.print(f"sanity loss: {loss.item():.4f}")

    # Train for a fixed number of optimizer steps.
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
    # Use shared profiler so SU/TTS metrics stay directly comparable.
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
                # Measure time spent waiting on the dataloader.
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_time = time.perf_counter() - data_wait_start
            else:
                batch = next(data_iter)
        except StopIteration:
            # Restart iterator on exhaustion and advance shuffle epoch.
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
            # Track conservative shard cursor across workers and ranks.
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

        with accelerator.accumulate(model):
            log_sample_ids(batch.get("sample_ids"))
            model_batch = _drop_non_model_fields(batch)
            with accelerator.autocast():
                outputs = model(**model_batch)
            loss = outputs.loss
            if loss is None:
                raise ValueError("loss is missing during training")

            labels = model_batch.get("labels")
            if labels is None:
                raise ValueError("labels are required for token-weighted loss logging")
            target_tokens_micro = (labels != IGNORE_INDEX).sum()
            # Accumulate token-weighted loss over micro-steps.
            loss_sum_local += loss.detach().float() * target_tokens_micro.to(dtype=torch.float32)
            target_tokens_local += target_tokens_micro

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                # Clip once per optimizer step.
                grad_norm = accelerator.clip_grad_norm_(optim_params, max_grad_norm)
                grad_norm_value = (
                    float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # Compute global token-weighted loss for this optimizer step.
                global_loss_sum = accelerator.reduce(loss_sum_local, reduction="sum")
                global_tokens = accelerator.reduce(target_tokens_local, reduction="sum")
                num_target_tokens = int(global_tokens.item())
                if num_target_tokens <= 0:
                    raise ValueError("global target token count is zero; cannot compute train/loss")
                train_loss = global_loss_sum.item() / float(num_target_tokens)
                logs = {
                    "train/loss": train_loss,
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

    # Always save final checkpoint.
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
