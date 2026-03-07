"""Stage-2 PomTalker: HF-native speech LM plus gate fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from .constants import SPEECH_VOCAB_SIZE
from .gate_fusion import GateFusion
from .tokenizers import TokenIds, build_pom_tokenizer, ensure_tokenizer_contract


class PomTalkerConfig(Qwen3Config):
    """Qwen3 config extended with Stage-2 speech and fusion metadata."""

    model_type = "pom_talker_qwen3"

    def __init__(
        self,
        *,
        speech_vocab_size: int = SPEECH_VOCAB_SIZE,
        text_vocab_size: Optional[int] = None,
        speech_token_offset: Optional[int] = None,
        llm_hidden_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Keep Stage-2 speech-token contract inside config for save/load parity.
        self.speech_vocab_size = max(0, int(speech_vocab_size))
        self.text_vocab_size = (
            None if text_vocab_size is None else max(0, int(text_vocab_size))
        )
        self.speech_token_offset = (
            None if speech_token_offset is None else int(speech_token_offset)
        )
        self.llm_hidden_dim = (
            int(llm_hidden_dim) if llm_hidden_dim is not None else int(self.hidden_size)
        )


class PomTalker(Qwen3ForCausalLM):
    """HF-native Stage-2 talker that owns speech LM weights and gate fusion."""

    config_class = PomTalkerConfig

    def __init__(self, config: PomTalkerConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        if int(config.llm_hidden_dim) <= 0:
            raise ValueError("llm_hidden_dim must be > 0")
        # Match gate output dim to current embedding dim of the talker speech LM.
        speech_embed_dim = int(self.get_input_embeddings().embedding_dim)
        self.gate_fusion = GateFusion(
            llm_hidden_dim=int(config.llm_hidden_dim),
            speech_embed_dim=speech_embed_dim,
        )

    @property
    def speech_vocab_size(self) -> int:
        """Return speech vocab size from config."""
        return int(self.config.speech_vocab_size)

    @property
    def text_vocab_size(self) -> Optional[int]:
        """Return base text vocab size used for speech-token offset mapping."""
        value = self.config.text_vocab_size
        return None if value is None else int(value)

    @property
    def speech_token_offset(self) -> Optional[int]:
        """Return speech-token offset inside the expanded LM vocabulary."""
        value = self.config.speech_token_offset
        return None if value is None else int(value)

    @property
    def total_vocab_size(self) -> int:
        """Return total vocabulary size after text+speech extension."""
        return int(self.config.vocab_size)

    def map_unit_ids(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """Map raw unit ids [0..speech_vocab_size-1] into LM token ids."""
        offset = self.speech_token_offset
        if offset is None or self.speech_vocab_size <= 0:
            raise ValueError("speech_vocab_size must be > 0 to map unit ids")
        if unit_ids.numel() == 0:
            return unit_ids
        if int(unit_ids.min().item()) < 0 or int(unit_ids.max().item()) >= self.speech_vocab_size:
            raise ValueError("unit ids must be in [0, speech_vocab_size-1]")
        return unit_ids + int(offset)

    def embed_content_ids(self, content_ids: torch.Tensor) -> torch.Tensor:
        """Embed one rank-1 content-id row using this talker's token embeddings."""
        if content_ids.ndim != 1:
            raise ValueError("content_ids must be rank-1")
        if int(content_ids.numel()) <= 0:
            raise ValueError("content_ids must be non-empty")

        # Keep lookup on the same device/dtype as the model embedding table.
        embed_tokens = self.get_input_embeddings()
        ids = content_ids.to(device=embed_tokens.weight.device, dtype=torch.long)
        return embed_tokens(ids)

    def fuse(
        self,
        *,
        hidden_rows: Sequence[torch.Tensor],
        content_ids: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Fuse one batch of aligned thinker hidden rows and content token ids."""
        if len(hidden_rows) != len(content_ids):
            raise ValueError("hidden_rows and content_ids must have the same batch size")

        fused_rows: list[torch.Tensor] = []
        for row_idx, (hidden_row, content_row) in enumerate(zip(hidden_rows, content_ids)):
            if hidden_row.ndim != 2:
                raise ValueError(f"hidden_rows[{row_idx}] must be rank-2 [tokens, hidden_dim]")
            if int(hidden_row.size(0)) <= 0:
                raise ValueError(f"hidden_rows[{row_idx}] must be non-empty")
            if int(hidden_row.size(1)) != int(self.config.llm_hidden_dim):
                raise ValueError(
                    f"hidden_rows[{row_idx}] dim {int(hidden_row.size(1))} != llm_hidden_dim {int(self.config.llm_hidden_dim)}"
                )

            text_embed_row = self.embed_content_ids(torch.as_tensor(content_row, dtype=torch.long))
            if int(text_embed_row.size(0)) != int(hidden_row.size(0)):
                raise ValueError(
                    f"sample {row_idx}: hidden/content token count mismatch "
                    f"({int(hidden_row.size(0))} vs {int(text_embed_row.size(0))})"
                )
            # Gate fusion blends thinker semantics with exact text-token identity.
            fused_rows.append(self.gate_fusion(hidden_row, text_embed_row))
        return fused_rows


AutoConfig.register(PomTalkerConfig.model_type, PomTalkerConfig)
AutoModelForCausalLM.register(PomTalkerConfig, PomTalker)


def _load_base_qwen_config(
    *,
    base_model_id: str,
    cache_dir: Optional[str],
) -> Qwen3Config:
    """Load and validate the base Qwen3 config."""
    base_config = AutoConfig.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
    )
    if not isinstance(base_config, Qwen3Config):
        raise TypeError(
            f"Expected Qwen3Config from {base_model_id}, got {type(base_config).__name__}"
        )
    return base_config


def _to_pom_talker_config(base_config: Union[Qwen3Config, Dict[str, Any]]) -> PomTalkerConfig:
    """Convert base Qwen config fields into PomTalkerConfig."""
    if isinstance(base_config, Qwen3Config):
        config_dict = base_config.to_dict()
    elif isinstance(base_config, dict):
        config_dict = dict(base_config)
    else:
        raise TypeError("base_config must be Qwen3Config or dict")
    # Drop base model_type so PomTalkerConfig can set its own type cleanly.
    config_dict.pop("model_type", None)
    return PomTalkerConfig(**config_dict)


def build_talker(
    *,
    llm_hidden_dim: Optional[int] = None,
    base_model_id: str = "Qwen/Qwen3-0.6B",
    base_cache_dir: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    speech_vocab_size: int = SPEECH_VOCAB_SIZE,
    tokenizer: Optional[AutoTokenizer] = None,
    token_ids: Optional[TokenIds] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PomTalker, AutoTokenizer, TokenIds]:
    """Build Stage-2 PomTalker and return shared tokenizer/token contract."""
    if tokenizer is None:
        tokenizer = build_pom_tokenizer(
            base_model_id=base_model_id,
            cache_dir=base_cache_dir,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    elif tokenizer_kwargs:
        raise ValueError("tokenizer_kwargs cannot be set when tokenizer is provided")

    token_ids = ensure_tokenizer_contract(tokenizer, token_ids)
    speech_vocab_size = max(0, int(speech_vocab_size))
    text_vocab_size = int(len(tokenizer))
    total_vocab_size = int(text_vocab_size + speech_vocab_size)
    speech_token_offset = text_vocab_size if speech_vocab_size > 0 else None

    # Start from base Qwen config and add Stage-2 talker metadata.
    config = _to_pom_talker_config(
        _load_base_qwen_config(base_model_id=base_model_id, cache_dir=base_cache_dir)
    )
    resolved_hidden_dim = int(config.hidden_size) if llm_hidden_dim is None else int(llm_hidden_dim)
    if resolved_hidden_dim <= 0:
        raise ValueError("llm_hidden_dim must be > 0")
    config.llm_hidden_dim = resolved_hidden_dim
    config.speech_vocab_size = speech_vocab_size
    config.text_vocab_size = text_vocab_size
    config.speech_token_offset = speech_token_offset

    load_kwargs: Dict[str, Any] = {"cache_dir": base_cache_dir}
    load_kwargs["torch_dtype"] = dtype if dtype is not None else "auto"
    talker = PomTalker.from_pretrained(
        base_model_id,
        config=config,
        **load_kwargs,
    )

    # Expand embeddings once so speech tokens are part of the trainable vocabulary.
    talker.resize_token_embeddings(total_vocab_size)
    if int(talker.get_input_embeddings().num_embeddings) != total_vocab_size:
        raise ValueError("PomTalker embedding size does not match tokenizer + speech vocab")
    return talker, tokenizer, token_ids


__all__ = ["PomTalker", "PomTalkerConfig", "build_talker"]
