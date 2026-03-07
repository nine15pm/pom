"""Speech-aware Hugging Face wrapper around Qwen3 — the Thinker."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

from .constants import IGNORE_INDEX
from .speech_mixin import SpeechMixin, SpeechMixinForCausalLM
from .speech_encoder import WhisperFeatureEncoder
from .speech_projector import FrameStackProjector
from .tokenizers import TokenIds, build_pom_tokenizer, ensure_tokenizer_contract


DEFAULT_SPEECH_ENCODER_ID = "openai/whisper-large-v3"
DEFAULT_FRAME_STACK = 5
DEFAULT_PROJECTOR_HIDDEN_DIM = 2048


class PomThinkerConfig(Qwen3Config):
    """Qwen3 configuration extended with speech metadata."""

    model_type = "pom_thinker_qwen3"

    def __init__(
        self,
        *,
        speech_token_id: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
        speech_enabled: bool = False,
        speech_encoder_config: Optional[Dict[str, Any]] = None,
        speech_projector_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.speech_token_id = speech_token_id
        self.ignore_index = ignore_index
        # Store speech-module metadata so HF artifacts are self-describing.
        self.speech_enabled = bool(speech_enabled)
        self.speech_encoder_config = (
            dict(speech_encoder_config) if speech_encoder_config is not None else None
        )
        self.speech_projector_config = (
            dict(speech_projector_config) if speech_projector_config is not None else None
        )


class PomThinkerModel(SpeechMixin, Qwen3Model):
    """Base Qwen3 transformer with speech mixin helpers attached."""

    config_class = PomThinkerConfig

    def __init__(self, config: PomThinkerConfig) -> None:  # type: ignore[override]
        super().__init__(config)


class PomThinker(SpeechMixinForCausalLM, Qwen3ForCausalLM):
    """Causal LM head that stitches speech features into the Qwen3 decoder."""

    config_class = PomThinkerConfig

    def __init__(self, config: PomThinkerConfig) -> None:  # type: ignore[override]
        super().__init__(config)
        # Rebuild speech modules from config so HF artifacts load end-to-end in one pass.
        encoder, projector = _build_speech_modules_from_config(
            config=config,
            target_dim=int(config.hidden_size),
        )
        self.set_speech_modules(speech_encoder=encoder, speech_projector=projector)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        speech_waveforms: Optional[Any] = None,
        speech_features: Optional[Any] = None,
        speech_sampling_rate: Optional[Union[int, Sequence[int]]] = None,
        **kwargs: Any,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech_waveforms=speech_waveforms,
                speech_features=speech_features,
                speech_sampling_rate=speech_sampling_rate,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


AutoConfig.register(PomThinkerConfig.model_type, PomThinkerConfig)
AutoModel.register(PomThinkerConfig, PomThinkerModel)
AutoModelForCausalLM.register(PomThinkerConfig, PomThinker)


def _load_base_qwen_config(
    *,
    base_model_id: str,
    cache_dir: Optional[str],
) -> Qwen3Config:
    """Load the base LM config using its native model_type."""
    base_config = AutoConfig.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
    )
    if not isinstance(base_config, Qwen3Config):
        raise TypeError(
            f"Expected Qwen3Config from {base_model_id}, got {type(base_config).__name__}"
        )
    return base_config


def _to_pom_thinker_config(base_config: Union[Qwen3Config, Dict[str, Any]]) -> PomThinkerConfig:
    """Convert base config fields into PomThinkerConfig."""
    if isinstance(base_config, Qwen3Config):
        config_dict = base_config.to_dict()
    elif isinstance(base_config, dict):
        config_dict = dict(base_config)
    else:
        raise TypeError("base_config must be Qwen3Config or dict")

    # Drop base model_type so PomThinkerConfig sets its own model_type cleanly.
    config_dict.pop("model_type", None)
    return PomThinkerConfig(**config_dict)


def _build_speech_modules(
    speech_spec: Optional[Dict[str, Any]],
    *,
    target_dim: int,
) -> Tuple[Optional[WhisperFeatureEncoder], Optional[FrameStackProjector]]:
    if speech_spec is None:
        return None, None
    if not isinstance(speech_spec, dict):
        raise TypeError("speech spec must be a dict of parameters")

    encoder_id = speech_spec.get("encoder_id", DEFAULT_SPEECH_ENCODER_ID)
    encoder_cache = speech_spec.get("encoder_cache", None)
    frame_stack = int(speech_spec.get("frame_stack", DEFAULT_FRAME_STACK))
    hidden_dim = int(speech_spec.get("projector_hidden_dim", DEFAULT_PROJECTOR_HIDDEN_DIM))

    encoder = WhisperFeatureEncoder(model_id=encoder_id, cache_dir=encoder_cache)
    projector = FrameStackProjector(
        frame_stack=frame_stack,
        input_dim=encoder.hidden_size,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
    )
    return encoder, projector


# Rebuild speech modules from serialized config metadata only.
def _build_speech_modules_from_config(
    *,
    config: PomThinkerConfig,
    target_dim: int,
) -> Tuple[Optional[WhisperFeatureEncoder], Optional[FrameStackProjector]]:
    if not bool(getattr(config, "speech_enabled", False)):
        return None, None

    encoder_cfg = getattr(config, "speech_encoder_config", None)
    projector_cfg = getattr(config, "speech_projector_config", None)
    if not isinstance(encoder_cfg, dict) or not isinstance(projector_cfg, dict):
        raise ValueError("speech_enabled=true requires speech_encoder_config and speech_projector_config")

    whisper_cfg = encoder_cfg.get("whisper_config")
    feature_extractor_cfg = encoder_cfg.get("feature_extractor_config")
    if not isinstance(whisper_cfg, dict) or not isinstance(feature_extractor_cfg, dict):
        raise ValueError("speech_encoder_config must include whisper_config and feature_extractor_config")

    encoder = WhisperFeatureEncoder.from_configs(
        whisper_config=whisper_cfg,
        feature_extractor_config=feature_extractor_cfg,
    )

    projector_target_dim = int(projector_cfg.get("target_dim", target_dim))
    if projector_target_dim != int(target_dim):
        raise ValueError(
            f"speech projector target_dim {projector_target_dim} does not match hidden_size {int(target_dim)}"
        )
    projector = FrameStackProjector(
        frame_stack=int(projector_cfg["frame_stack"]),
        input_dim=int(projector_cfg["input_dim"]),
        target_dim=int(target_dim),
        hidden_dim=int(projector_cfg["hidden_dim"]),
    )
    return encoder, projector


# Persist speech build metadata on Thinker config for artifact reload.
def _store_speech_metadata_on_config(
    *,
    config: PomThinkerConfig,
    speech_encoder: Optional[WhisperFeatureEncoder],
    speech_projector: Optional[FrameStackProjector],
) -> None:
    if speech_encoder is None or speech_projector is None:
        config.speech_enabled = False
        config.speech_encoder_config = None
        config.speech_projector_config = None
        return
    config.speech_enabled = True
    config.speech_encoder_config = speech_encoder.to_config_dict()
    config.speech_projector_config = speech_projector.to_config_dict()


def build_thinker(
    *,
    base_model_id: str = "Qwen/Qwen3-0.6B",
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    speech: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    token_ids: Optional[TokenIds] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[PomThinker, AutoTokenizer, TokenIds]:
    """Load Qwen3 with pom speech defaults applied."""

    # Load native base config first, then map to our speech-extended config.
    base_config = _load_base_qwen_config(
        base_model_id=base_model_id,
        cache_dir=cache_dir,
    )
    config = _to_pom_thinker_config(base_config)
    config.ignore_index = IGNORE_INDEX
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    if tokenizer is None:
        tokenizer = build_pom_tokenizer(
            base_model_id=base_model_id,
            cache_dir=cache_dir,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    elif tokenizer_kwargs:
        raise ValueError("tokenizer_kwargs cannot be set when tokenizer is provided")

    token_ids = ensure_tokenizer_contract(tokenizer, token_ids)
    config.speech_token_id = token_ids.speech_id

    load_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if model_kwargs:
        load_kwargs.update(model_kwargs)

    model = PomThinker.from_pretrained(
        base_model_id,
        config=config,
        **load_kwargs,
    )
    model.resize_token_embeddings(len(tokenizer))

    encoder, projector = _build_speech_modules(
        speech,
        target_dim=model.config.hidden_size,
    )
    model.set_speech_modules(speech_encoder=encoder, speech_projector=projector)
    _store_speech_metadata_on_config(
        config=model.config,
        speech_encoder=encoder,
        speech_projector=projector,
    )

    return model, tokenizer, token_ids


__all__ = [
    "PomThinkerConfig",
    "PomThinkerModel",
    "PomThinker",
    "build_thinker",
]
