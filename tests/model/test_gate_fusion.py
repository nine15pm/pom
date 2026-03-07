from __future__ import annotations

import pytest
import torch

from model.gate_fusion import GateFusion


DEVICE = torch.device("cuda")


def test_forward_contract_and_gate_bounds():
    # Validate the public tensor contract used by S2S training code.
    module = GateFusion(llm_hidden_dim=8, speech_embed_dim=6).to(device=DEVICE, dtype=torch.float32)
    expected_device = next(module.parameters()).device

    # Use mixed input dtypes to confirm GateFusion normalizes to module dtype/device.
    hidden_states = torch.randn(2, 3, 8, device=DEVICE, dtype=torch.float64)
    text_embeddings = torch.randn(2, 3, 6, device=DEVICE, dtype=torch.float16)

    fused, gate = module(hidden_states, text_embeddings, return_gate=True)

    # Fused outputs should align with speech embedding axes.
    assert fused.shape == (2, 3, 6)
    assert gate.shape == (2, 3, 6)
    assert fused.device == expected_device
    assert gate.device == expected_device
    assert fused.dtype == torch.float32
    assert gate.dtype == torch.float32

    # Sigmoid gates must stay in [0, 1], and outputs should stay numerically valid.
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(gate).all()


def test_backward_updates_fusion_and_text_embedding_paths():
    # Ensure Stage-2 trainable paths receive gradients with frozen Thinker features.
    torch.manual_seed(7)
    module = GateFusion(llm_hidden_dim=8, speech_embed_dim=6).to(device=DEVICE, dtype=torch.float32)
    token_embed = torch.nn.Embedding(13, 6).to(device=DEVICE, dtype=torch.float32)

    # Hidden states emulate frozen Thinker outputs in S2S.
    hidden_states = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32, requires_grad=False)
    content_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=DEVICE, dtype=torch.long)
    text_embeddings = token_embed(content_ids)

    fused = module(hidden_states, text_embeddings)
    loss = fused.square().mean()
    loss.backward()

    # GateFusion parameters must receive training signal.
    fusion_grads = [param.grad for param in module.parameters()]
    assert all(grad is not None for grad in fusion_grads)
    assert any(torch.any(grad != 0) for grad in fusion_grads)

    # Speech LM text embedding path must also receive training signal.
    assert token_embed.weight.grad is not None
    assert torch.any(token_embed.weight.grad != 0)

    # Frozen hidden-state inputs should not accumulate gradients.
    assert hidden_states.grad is None


def test_fused_output_changes_when_either_input_changes():
    # Confirm fusion is not a one-branch pass-through by probing both input paths.
    torch.manual_seed(11)
    module = GateFusion(llm_hidden_dim=8, speech_embed_dim=6).to(device=DEVICE, dtype=torch.float32)

    hidden_a = torch.randn(2, 3, 8, device=DEVICE, dtype=torch.float32)
    hidden_b = hidden_a + 0.25
    text_a = torch.randn(2, 3, 6, device=DEVICE, dtype=torch.float32)
    text_b = text_a * 1.5

    out_aa = module(hidden_a, text_a)
    out_ba = module(hidden_b, text_a)
    out_ab = module(hidden_a, text_b)

    # Changing hidden states with fixed text should change the fused outputs.
    assert not torch.allclose(out_aa, out_ba)
    # Changing text embeddings with fixed hidden states should change outputs too.
    assert not torch.allclose(out_aa, out_ab)


def test_forward_raises_on_alignment_or_dimension_mismatch():
    # Fail fast on shape/dim mismatches so bad training batches do not silently pass through.
    module = GateFusion(llm_hidden_dim=8, speech_embed_dim=6).to(device=DEVICE, dtype=torch.float32)

    good_hidden = torch.randn(2, 3, 8, device=DEVICE, dtype=torch.float32)
    good_text = torch.randn(2, 3, 6, device=DEVICE, dtype=torch.float32)

    # Rank mismatch between hidden and text tensors.
    with pytest.raises(ValueError):
        module(good_hidden.unsqueeze(0), good_text)

    # Token-axis mismatch: leading dimensions must align.
    with pytest.raises(ValueError):
        module(good_hidden, torch.randn(2, 4, 6, device=DEVICE, dtype=torch.float32))

    # Hidden-state feature width must match configured llm_hidden_dim.
    with pytest.raises(ValueError):
        module(torch.randn(2, 3, 7, device=DEVICE, dtype=torch.float32), good_text)

    # Text embedding width must match configured speech_embed_dim.
    with pytest.raises(ValueError):
        module(good_hidden, torch.randn(2, 3, 5, device=DEVICE, dtype=torch.float32))
