"""Runtime loading helpers for Phase A inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from cosyvoice2.speech_decoder import SPEECH_TOKEN_MAX, SPEECH_TOKEN_MIN, SpeechDecoder
from inference.config import InferenceConfig
from model.pom_talker import PomTalker, build_talker
from model.pom_thinker import PomThinker, build_thinker
from model.tokenizers import TokenIds, build_pom_tokenizer, resolve_token_ids

@dataclass(frozen=True)
class RuntimeTokenContract:
    """Resolved tokenizer ids required by the inference runtime contract."""

    speech_id: int
    sep_id: int
    eos_id: int
    assistant_stop_id: int


@dataclass(frozen=True)
class RuntimeBundle:
    """All startup-loaded runtime components reused across inference requests."""

    thinker: Any
    talker: Any
    tokenizer: Any
    token_ids: TokenIds
    token_contract: RuntimeTokenContract
    decoder: Any | None
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class ModelCachePaths:
    """Resolved model-cache layout under one canonical cache root."""

    root: Path
    thinker: Path
    talker: Path
    tokenizer: Path
    decoder: Path


def _resolve_model_cache_paths(cfg: InferenceConfig) -> ModelCachePaths:
    """Resolve one cache root and the fixed component subdirectories."""
    root = Path(str(cfg["models"]["cache_dir"]))
    if not root.is_dir():
        raise FileNotFoundError(f"models.cache_dir not found: {root}")
    return ModelCachePaths(
        root=root,
        thinker=root / "thinker",
        talker=root / "talker",
        tokenizer=root / "tokenizer",
        decoder=root / "decoder",
    )


def _resolve_tokenizer_source(
    *,
    artifact_mode: str,
    tokenizer_override: str | None,
    checkpoint_cfg: dict[str, Any] | None,
    cache_paths: ModelCachePaths,
) -> str:
    """Resolve tokenizer source from explicit config contract."""
    override = tokenizer_override
    if override is not None:
        if artifact_mode == "hf":
            raise ValueError("tokenizer.source is only valid when models.artifact_mode='checkpoint'")
        return str(override)
    if artifact_mode == "checkpoint":
        if checkpoint_cfg is None:
            raise ValueError("models.checkpoint must be set when models.artifact_mode='checkpoint'")
        return str(checkpoint_cfg["base_model_id"])
    return str(cache_paths.tokenizer)


def _token_id_from_token(tokenizer: Any, token: str, *, field_name: str) -> int:
    """Resolve one token id and reject missing/unk mappings."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"tokenizer is missing {field_name} token: {token!r}")
    unk_id = tokenizer.unk_token_id
    if unk_id is not None and int(token_id) == int(unk_id):
        raise ValueError(f"tokenizer mapped {field_name} token to unk: {token!r}")
    return int(token_id)


def _resolve_default_assistant_stop_id(tokenizer: Any) -> int:
    """Resolve default assistant stop token id from tokenizer eos_token_id."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("tokenizer must define eos_token_id")
    return int(eos_id)


def _resolve_assistant_stop_id(tokenizer: Any, stop_token_override: str | None) -> int:
    """Resolve assistant stop token id using optional config override."""
    if stop_token_override is None:
        return _resolve_default_assistant_stop_id(tokenizer)
    return _token_id_from_token(
        tokenizer,
        str(stop_token_override),
        field_name="assistant_stop_token",
    )


def _resolve_token_contract(
    tokenizer: Any,
    *,
    stop_token_override: str | None,
) -> tuple[TokenIds, RuntimeTokenContract]:
    """Resolve all required tokenizer ids once at startup."""
    token_ids = resolve_token_ids(tokenizer)

    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("tokenizer must define eos_token_id")

    return token_ids, RuntimeTokenContract(
        speech_id=int(token_ids.speech_id),
        sep_id=int(token_ids.sep_id),
        eos_id=int(eos_id),
        assistant_stop_id=_resolve_assistant_stop_id(tokenizer, stop_token_override),
    )


def _load_strict_checkpoint_state(path_or_dir: str, *, name: str) -> dict[str, torch.Tensor]:
    """Load one strict checkpoint state from one path with one .bin weight file."""
    path = Path(path_or_dir)
    if path.is_dir():
        bin_file = path / "pytorch_model.bin"
        if not bin_file.is_file():
            raise FileNotFoundError(f"{name} checkpoint is missing pytorch_model.bin: {path}")
        # Keep fallback loading deterministic: checkpoint dirs must not contain shard files.
        extra_bins = sorted(path.glob("pytorch_model*.bin"))
        if len(extra_bins) != 1 or extra_bins[0].name != "pytorch_model.bin":
            raise ValueError(
                f"{name} checkpoint dir must contain exactly one model file: pytorch_model.bin"
            )
        model_path = bin_file
    else:
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint path not found: {path}")
        if path.name != "pytorch_model.bin":
            raise ValueError(f"{name} checkpoint file must be named pytorch_model.bin")
        model_path = path

    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"{name} checkpoint must be a state_dict mapping: {model_path}")
    return state


def _load_hf_thinker_strict(
    *,
    thinker_path: str,
    cache_dir: str | None,
    dtype: torch.dtype,
) -> PomThinker:
    """Load HF Thinker and reject any state-dict key mismatch."""
    thinker, loading_info = PomThinker.from_pretrained(
        thinker_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        output_loading_info=True,
    )
    missing = sorted(str(key) for key in loading_info.get("missing_keys", []))
    unexpected = sorted(str(key) for key in loading_info.get("unexpected_keys", []))
    if missing or unexpected:
        raise ValueError(
            "thinker HF load key mismatch: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
    return thinker


def _load_hf_talker_strict(
    *,
    talker_path: str,
    cache_dir: str | None,
    dtype: torch.dtype,
) -> PomTalker:
    """Load HF Talker and reject any state-dict key mismatch."""
    talker, loading_info = PomTalker.from_pretrained(
        talker_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        output_loading_info=True,
    )
    missing = sorted(str(key) for key in loading_info.get("missing_keys", []))
    unexpected = sorted(str(key) for key in loading_info.get("unexpected_keys", []))
    if missing or unexpected:
        raise ValueError(
            "talker HF load key mismatch: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )
    return talker


def _validate_runtime_contract(
    *,
    thinker: PomThinker,
    talker: PomTalker,
    tokenizer: PreTrainedTokenizer,
    token_ids: TokenIds,
    speech_vocab_size: int,
) -> None:
    """Validate cross-model/tokenizer assumptions once at startup."""
    if int(getattr(thinker.config, "speech_token_id", -1)) != int(token_ids.speech_id):
        raise ValueError("thinker speech_token_id does not match tokenizer <speech> id")
    # Inference always consumes user audio, so speech modules must be present.
    if thinker.get_speech_encoder() is None or thinker.get_speech_projector() is None:
        raise ValueError("thinker speech encoder/projector must be initialized for inference")

    text_vocab_size = int(len(tokenizer))
    if talker.text_vocab_size != text_vocab_size:
        raise ValueError("talker text_vocab_size does not match tokenizer length")
    if talker.speech_token_offset != text_vocab_size:
        raise ValueError("talker speech_token_offset must equal tokenizer length")
    if talker.speech_vocab_size != int(speech_vocab_size):
        raise ValueError("talker speech_vocab_size does not match models.speech_vocab_size")
    if talker.total_vocab_size != text_vocab_size + talker.speech_vocab_size:
        raise ValueError("talker total vocab size is inconsistent with text+speech vocab sizes")
    if int(talker.get_input_embeddings().num_embeddings) != int(talker.total_vocab_size):
        raise ValueError("talker embedding size does not match total vocab size")

    thinker_hidden = int(thinker.config.hidden_size)
    talker_hidden = int(talker.config.llm_hidden_dim)
    if thinker_hidden != talker_hidden:
        raise ValueError("thinker hidden_size must equal talker llm_hidden_dim")


def load_runtime(cfg: InferenceConfig, *, load_decoder: bool = True) -> RuntimeBundle:
    """Load inference runtime; optionally skip decoder assets for generation-only workers."""
    # Keep inference runtime fixed to bf16 on CUDA for a single clean path.
    if str(cfg["runtime"]["dtype"]).lower() != "bf16":
        raise ValueError("runtime.dtype must be 'bf16' for inference runtime")
    if str(cfg["runtime"]["device"]).lower() != "cuda":
        raise ValueError("runtime.device must be 'cuda' for inference runtime")
    dtype = torch.bfloat16
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("runtime.device is 'cuda' but CUDA is not available")
    models_cfg = cfg["models"]
    artifact_mode = str(models_cfg["artifact_mode"])
    cache_paths = _resolve_model_cache_paths(cfg)
    seed = int(cfg["runtime"]["seed"])

    # Apply startup seed once so decode behavior is reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Build one shared tokenizer/token contract used by both Thinker and Talker.
    if artifact_mode == "checkpoint":
        # Checkpoint fallback must mirror builder behavior and register Pom special tokens.
        tokenizer = build_pom_tokenizer(
            base_model_id=_resolve_tokenizer_source(
                artifact_mode=artifact_mode,
                tokenizer_override=cfg["tokenizer"]["source"],
                checkpoint_cfg=models_cfg["checkpoint"],
                cache_paths=cache_paths,
            ),
            cache_dir=models_cfg["base_cache_dir"],
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            _resolve_tokenizer_source(
                artifact_mode=artifact_mode,
                tokenizer_override=cfg["tokenizer"]["source"],
                checkpoint_cfg=models_cfg["checkpoint"],
                cache_paths=cache_paths,
            ),
            cache_dir=models_cfg["base_cache_dir"],
        )
    token_ids, token_contract = _resolve_token_contract(
        tokenizer,
        stop_token_override=cfg["tokenizer"]["assistant_stop_token"],
    )

    if artifact_mode == "hf":
        thinker = _load_hf_thinker_strict(
            thinker_path=str(cache_paths.thinker),
            cache_dir=models_cfg["base_cache_dir"],
            dtype=dtype,
        )
        talker = _load_hf_talker_strict(
            talker_path=str(cache_paths.talker),
            cache_dir=models_cfg["base_cache_dir"],
            dtype=dtype,
        )
    elif artifact_mode == "checkpoint":
        checkpoint_cfg = models_cfg["checkpoint"]
        thinker, _, _ = build_thinker(
            base_model_id=str(checkpoint_cfg["base_model_id"]),
            cache_dir=models_cfg["base_cache_dir"],
            torch_dtype=dtype,
            speech={
                "encoder_id": str(checkpoint_cfg["speech_encoder_id"]),
                "encoder_cache": models_cfg["speech_encoder_cache_dir"],
                "frame_stack": int(checkpoint_cfg["frame_stack"]),
                "projector_hidden_dim": int(checkpoint_cfg["projector_hidden_dim"]),
            },
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        talker, _, _ = build_talker(
            llm_hidden_dim=int(thinker.config.hidden_size),
            base_model_id=str(checkpoint_cfg["base_model_id"]),
            base_cache_dir=models_cfg["base_cache_dir"],
            dtype=dtype,
            speech_vocab_size=int(models_cfg["speech_vocab_size"]),
            tokenizer=tokenizer,
            token_ids=token_ids,
        )

        thinker.load_state_dict(
            _load_strict_checkpoint_state(str(cache_paths.thinker), name="thinker"),
            strict=True,
        )
        talker.load_state_dict(
            _load_strict_checkpoint_state(str(cache_paths.talker), name="talker"),
            strict=True,
        )
    else:
        raise ValueError(f"unsupported artifact mode: {artifact_mode}")

    _validate_runtime_contract(
        thinker=thinker,
        talker=talker,
        tokenizer=tokenizer,
        token_ids=token_ids,
        speech_vocab_size=int(models_cfg["speech_vocab_size"]),
    )

    # Keep Phase A runtime in inference mode and prevent accidental Thinker training.
    for param in thinker.parameters():
        param.requires_grad = False
    thinker.eval()
    talker.eval()
    thinker.to(device=device, dtype=dtype)
    talker.to(device=device, dtype=dtype)

    decoder = None
    # Keep decoder loading optional so generation and decode stages stay cleanly separated.
    if bool(load_decoder) and cache_paths.decoder.is_dir():
        # Decoder supports raw unit ids in [0, 6560], so vocab must be exactly 6561.
        decoder_vocab_size = int(SPEECH_TOKEN_MAX - SPEECH_TOKEN_MIN + 1)
        if int(talker.speech_vocab_size) != decoder_vocab_size:
            raise ValueError(
                "decoder contract mismatch: talker speech_vocab_size must match decoder token range"
            )
        decoder = SpeechDecoder(str(cache_paths.decoder), device=device)

    return RuntimeBundle(
        thinker=thinker,
        talker=talker,
        tokenizer=tokenizer,
        token_ids=token_ids,
        token_contract=token_contract,
        decoder=decoder,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "RuntimeBundle",
    "RuntimeTokenContract",
    "load_runtime",
]
