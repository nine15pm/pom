"""Speech LM (PomTTS) with Hugging Face-native config/model packaging."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from .constants import SPEECH_VOCAB_SIZE
from .tokenizers import TokenIds, build_pom_tokenizer, ensure_tokenizer_contract


class PomTTSConfig(Qwen3Config):
    """Qwen3 config extended with Stage-1b speech-token metadata."""

    model_type = "pom_tts_qwen3"

    def __init__(
        self,
        *,
        speech_vocab_size: int = SPEECH_VOCAB_SIZE,
        text_vocab_size: Optional[int] = None,
        speech_token_offset: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Store speech-token contract directly in config for clean save/load behavior.
        self.speech_vocab_size = max(0, int(speech_vocab_size))
        self.text_vocab_size = (
            None if text_vocab_size is None else max(0, int(text_vocab_size))
        )
        self.speech_token_offset = (
            None if speech_token_offset is None else int(speech_token_offset)
        )


class PomTTS(Qwen3ForCausalLM):
    """Qwen3 causal LM with Pom speech-token mapping helpers."""

    config_class = PomTTSConfig

    def __init__(self, config: PomTTSConfig) -> None:  # type: ignore[override]
        super().__init__(config)

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


AutoConfig.register(PomTTSConfig.model_type, PomTTSConfig)
AutoModelForCausalLM.register(PomTTSConfig, PomTTS)


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


def _to_pom_tts_config(base_config: Union[Qwen3Config, Dict[str, Any]]) -> PomTTSConfig:
    """Convert base Qwen config fields into PomTTSConfig."""
    if isinstance(base_config, Qwen3Config):
        config_dict = base_config.to_dict()
    elif isinstance(base_config, dict):
        config_dict = dict(base_config)
    else:
        raise TypeError("base_config must be Qwen3Config or dict")
    # Drop base model_type so PomTTSConfig can set its own type cleanly.
    config_dict.pop("model_type", None)
    return PomTTSConfig(**config_dict)


def build_pom_tts(
    *,
    base_model_id: str = "Qwen/Qwen3-0.6B",
    base_cache_dir: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    speech_vocab_size: int = SPEECH_VOCAB_SIZE,
    tokenizer: Optional[AutoTokenizer] = None,
    token_ids: Optional[TokenIds] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PomTTS, AutoTokenizer, TokenIds]:
    """Build PomTTS and tokenizer with one validated token-id contract."""
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

    # Start from base Qwen config and add Pom token metadata.
    config = _to_pom_tts_config(
        _load_base_qwen_config(base_model_id=base_model_id, cache_dir=base_cache_dir)
    )
    config.speech_vocab_size = speech_vocab_size
    config.text_vocab_size = text_vocab_size
    config.speech_token_offset = speech_token_offset

    load_kwargs: Dict[str, Any] = {"cache_dir": base_cache_dir}
    load_kwargs["torch_dtype"] = dtype if dtype is not None else "auto"
    tts = PomTTS.from_pretrained(
        base_model_id,
        config=config,
        **load_kwargs,
    )

    # Expand embeddings once so speech tokens are part of the trainable vocabulary.
    tts.resize_token_embeddings(total_vocab_size)
    if int(tts.get_input_embeddings().num_embeddings) != total_vocab_size:
        raise ValueError("PomTTS embedding size does not match tokenizer + speech vocab")
    return tts, tokenizer, token_ids


__all__ = ["PomTTS", "PomTTSConfig", "build_pom_tts"]
