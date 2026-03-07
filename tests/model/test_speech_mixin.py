from __future__ import annotations

import torch
from torch import nn

from model.constants import IGNORE_INDEX
from model.speech_mixin import SpeechMixinForCausalLM, SpeechMixin


DEVICE = torch.device("cuda")


class _DummyConfig:
    def __init__(self, vocab_size: int, embed_dim: int, speech_token_id: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = embed_dim
        self.speech_token_id = speech_token_id
        self.ignore_index = IGNORE_INDEX


class _Base(nn.Module):
    def __init__(self, config: _DummyConfig) -> None:
        super().__init__()
        self.config = config


class _DummyModel(SpeechMixinForCausalLM, _Base):
    def __init__(self, config: _DummyConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def get_input_embeddings(self):  # type: ignore[override]
        return self.embed_tokens


def _build_dummy():
    vocab_size = 10
    embed_dim = 6
    speech_id = 3
    cfg = _DummyConfig(vocab_size, embed_dim, speech_id)
    model = _DummyModel(cfg).to(device=DEVICE)
    return model, speech_id


def test_fusion_single_speech_with_mask_and_trim():
    model, speech_id = _build_dummy()

    # Batch of 1: tokens [X, speech, Y], but attention drops the first token
    input_ids = torch.tensor([[9, speech_id, 1]], device=DEVICE)
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool, device=DEVICE)
    labels = torch.tensor([[5, 6, 7]], device=DEVICE)

    # Speech features: 3 frames, mask trims last frame
    speech_feat = torch.randn(3, model.embed_tokens.embedding_dim, device=DEVICE)
    speech_mask = torch.tensor([1, 1, 0], dtype=torch.bool, device=DEVICE)

    out = model.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        speech_features=[(speech_feat, speech_mask)],
    )

    out_input_ids, pos_ids, attn, _, embeds, out_labels = out
    assert out_input_ids is None
    # valid text tokens: 1, valid speech frames: 2 -> total len 3
    assert embeds.shape[1] == 3
    assert torch.all(attn[0, :3]) and attn[0, 3:].sum() == 0
    assert torch.equal(pos_ids[0, :3], torch.tensor([0, 1, 2], device=DEVICE))
    # Speech frames ignored; trailing text token keeps its label
    assert torch.all(out_labels[0, 0:2] == IGNORE_INDEX)
    assert out_labels[0, 2].item() == labels[0, 2].item()  # token "1"


def test_fusion_two_speech_segments_order_and_labels():
    model, speech_id = _build_dummy()

    input_ids = torch.tensor([[speech_id, 4, speech_id, 5]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels = torch.tensor([[10, 11, 12, 13]], device=DEVICE)

    s1 = torch.randn(2, model.embed_tokens.embedding_dim, device=DEVICE)
    s2 = torch.randn(1, model.embed_tokens.embedding_dim, device=DEVICE)

    out = model.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels,
        speech_features=[s1, s2],
    )

    _, _, attn, _, embeds, out_labels = out
    # Order: speech1 (2), token 4 (1), speech2 (1), token 5 (1) -> len 5
    assert embeds.shape[1] == 5
    assert torch.all(attn)
    # Labels: speech -> ignore, text tokens preserved
    assert torch.all(out_labels[0, 0:2] == IGNORE_INDEX)
    assert out_labels[0, 2].item() == labels[0, 1].item()
    assert out_labels[0, 3] == IGNORE_INDEX
    assert out_labels[0, 4].item() == labels[0, 3].item()


def test_fusion_raises_on_dim_mismatch():
    model, speech_id = _build_dummy()
    wrong_dim_feat = torch.randn(2, model.embed_tokens.embedding_dim + 1, device=DEVICE)

    input_ids = torch.tensor([[speech_id]], device=DEVICE)
    attn = torch.tensor([[1]], dtype=torch.bool, device=DEVICE)

    try:
        model.prepare_inputs_labels_for_speech_and_text(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attn,
            past_key_values=None,
            labels=None,
            speech_features=[wrong_dim_feat],
        )
    except ValueError as exc:
        assert "speech feature dim" in str(exc)
    else:
        raise AssertionError("Expected ValueError for dim mismatch")


def test_fusion_raises_on_segment_count_mismatch():
    model, speech_id = _build_dummy()

    input_ids = torch.tensor([[speech_id, 2, speech_id]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)

    speech_feat = torch.randn(2, model.embed_tokens.embedding_dim, device=DEVICE)

    try:
        model.prepare_inputs_labels_for_speech_and_text(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            speech_features=[speech_feat],
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for speech segment count mismatch")


def test_fusion_loss_ignores_speech_labels():
    model, speech_id = _build_dummy()

    input_ids = torch.tensor([[speech_id, 4]], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    labels_a = torch.tensor([[111, 222]], device=DEVICE)
    labels_b = torch.tensor([[999, 222]], device=DEVICE)

    speech_feat = torch.randn(2, model.embed_tokens.embedding_dim, device=DEVICE)

    out_a = model.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels_a,
        speech_features=[speech_feat],
    )
    out_b = model.prepare_inputs_labels_for_speech_and_text(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels_b,
        speech_features=[speech_feat],
    )

    _, _, _, _, _, labels_out_a = out_a
    _, _, _, _, _, labels_out_b = out_b

    assert labels_out_a is not None and labels_out_b is not None
    assert torch.equal(labels_out_a, labels_out_b)
