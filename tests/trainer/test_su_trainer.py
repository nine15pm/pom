import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from model.pom_thinker import build_thinker
from train.su_trainer import _resolve_resume_dir, _validate_resume_compat


# Resolve the repo root for subprocess calls.
def _repo_root() -> Path:
    """Return the project root for subprocess execution."""
    # Walk upward from this file until we find the repo marker.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # If we fall through, the test environment is misconfigured.
    raise RuntimeError("Could not locate repo root (pyproject.toml not found)")


def _write_json_list(path: Path, records: list[dict]) -> None:
    """Write a JSON list file to disk."""
    path.write_text(json.dumps(records), encoding="utf-8")


def _fixtures_dir() -> Path:
    """Return the shared test fixtures directory."""
    return Path(__file__).resolve().parents[1] / "fixtures"


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    """Copy a fixture WAV into the temp audio root."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _write_config(path: Path, data: dict) -> None:
    """Serialize a YAML config for the trainer."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    """Load a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _run_su_trainer(
    config_path: Path,
    *,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run train.su_trainer and return the captured subprocess result."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "train.su_trainer", "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )


def _step_checkpoint_dirs(output_dir: Path) -> list[Path]:
    """Return all step checkpoint dirs sorted by step number."""
    checkpoint_root = output_dir / "checkpoints"
    return sorted(path for path in checkpoint_root.glob("step-*") if path.is_dir())


def _load_model_state(checkpoint_dir: Path) -> dict:
    """Load model weights from one per-step checkpoint directory."""
    bin_path = checkpoint_dir / "pytorch_model.bin"
    safe_path = checkpoint_dir / "model.safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    if safe_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("safetensors is required to load model.safetensors") from exc
        return load_file(str(safe_path))
    raise AssertionError(f"No model checkpoint found in {checkpoint_dir}")


def _build_trainer_cfg(
    *,
    model_id: str,
    whisper_id: str,
    json_path: Path,
    audio_root: Path,
    output_dir: Path,
    steps: int,
    batch_size: int = 1,
    grad_accum: int = 1,
    precision: str = "no",
    num_workers: int = 0,
    seed: int = 123,
    lr: float = 5.0e-5,
    save_every: int | None = None,
    keep_last_n_checkpoints: int | None = None,
    warmup_ratio: float | None = None,
    wandb: dict | None = None,
) -> dict:
    """Build a minimal SU trainer config used by integration tests."""
    # Keep defaults centralized so tests only specify fields that matter.
    training_cfg = {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "precision": precision,
        "num_workers": num_workers,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "output_dir": str(output_dir),
    }
    # Add optional training knobs only when a test needs them.
    if save_every is not None:
        training_cfg["save_every"] = save_every
    if keep_last_n_checkpoints is not None:
        training_cfg["keep_last_n_checkpoints"] = keep_last_n_checkpoints
    if warmup_ratio is not None:
        training_cfg["warmup_ratio"] = warmup_ratio

    return {
        "model": {
            "id": model_id,
            "speech_encoder_id": whisper_id,
            "frame_stack": 5,
            "adapter_hidden_dim": 2048,
        },
        "data": {
            "json_path": str(json_path),
            "audio_root": str(audio_root),
        },
        "training": training_cfg,
        "wandb": {"enabled": False} if wandb is None else wandb,
    }


def test_su_trainer_smoke_and_updates_trainables(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Single run should pass smoke checks and update only trainable modules."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    record = {
        "id": "conv-smoke-and-updates",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    # Capture baseline encoder + adapter weights before training.
    torch.manual_seed(123)
    baseline_model, baseline_tokenizer, _ = build_thinker(
        base_model_id=local_base_model_id,
        speech={
            "encoder_id": local_whisper_tiny_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    )
    encoder = baseline_model.get_speech_encoder()
    assert encoder is not None
    pre_encoder = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}

    projector = baseline_model.get_speech_projector()
    assert projector is not None
    pre_projector = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}

    reply_ids = baseline_tokenizer("a1", add_special_tokens=False)["input_ids"]
    assert reply_ids, "Tokenizer returned empty ids for test reply"
    output_embeddings = baseline_model.get_output_embeddings()
    assert output_embeddings is not None
    pre_reply_rows = output_embeddings.weight.detach().cpu()[reply_ids].clone()

    output_dir = tmp_path / "out"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=output_dir,
        # Warmup scheduler initializes lr=0 at step 1, so use 2 steps to see a real update.
        steps=2,
    )
    # Keep clipping explicit here so compat metadata contract is exercised.
    cfg["training"]["max_grad_norm"] = 0.7
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    # Use the shared subprocess helper so trainer invocation stays consistent.
    result = _run_su_trainer(cfg_path)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output
    assert "sanity loss:" in output

    # Read the terminal checkpoint produced by the 2-step run.
    checkpoint_dir = output_dir / "checkpoints" / "step-00000002"

    # The terminal step checkpoint should include core accelerate artifacts.
    checkpoint_files = [
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "optimizer.bin",
    ]
    assert any(path.exists() for path in checkpoint_files)

    # Resume metadata should match the expected terminal state contract.
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    assert trainer_state_path.exists()
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    assert trainer_state["global_step"] == 2
    assert trainer_state["grad_accum"] == 1
    assert trainer_state["micro_batches_seen"] >= 2
    assert "wandb_run_id" in trainer_state
    assert trainer_state["compat"]["max_grad_norm"] == 0.7

    # Load the checkpoint into a fresh model for comparison.
    trained_model, _, _ = build_thinker(
        base_model_id=local_base_model_id,
        speech={
            "encoder_id": local_whisper_tiny_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    )
    state = _load_model_state(checkpoint_dir)
    trained_model.load_state_dict(state, strict=True)

    # Encoder weights must be identical (frozen).
    trained_encoder = trained_model.get_speech_encoder()
    assert trained_encoder is not None
    for name, tensor in trained_encoder.state_dict().items():
        assert torch.equal(pre_encoder[name], tensor.cpu())

    # Adapter weights should change after one optimizer step.
    trained_projector = trained_model.get_speech_projector()
    assert trained_projector is not None
    projector_changed = False
    for name, tensor in trained_projector.state_dict().items():
        if not torch.equal(pre_projector[name], tensor.cpu()):
            projector_changed = True
            break
    assert projector_changed, "Adapter weights did not change after training"

    # LLM output embedding rows for the reply tokens should change too.
    trained_output_embeddings = trained_model.get_output_embeddings()
    assert trained_output_embeddings is not None
    post_reply_rows = trained_output_embeddings.weight.detach().cpu()[reply_ids]
    assert not torch.equal(pre_reply_rows, post_reply_rows)


def test_su_trainer_fails_on_empty_dataset(
    tmp_path,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Fail fast when the dataset yields zero samples."""
    audio_root = tmp_path / "data"
    audio_root.mkdir(parents=True, exist_ok=True)

    record = {
        "id": "conv-empty",
        "conversations": [
            {"from": "user", "value": "u1", "audio": None},
            {"from": "assistant", "value": "a1"},
        ],
    }
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    output_dir = tmp_path / "out"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=output_dir,
        steps=1,
    )
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    # Use the shared subprocess helper so trainer invocation stays consistent.
    result = _run_su_trainer(cfg_path)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "dataset produced no samples" in output


@pytest.mark.parametrize(
    ("keep_last_n", "expected_steps"),
    [
        (0, ["step-00000002"]),
    ],
)
def test_su_trainer_checkpoint_retention_policy(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
    keep_last_n,
    expected_steps,
):
    """Keep only the expected newest checkpoints for retention edge semantics."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    # Keep enough training pairs for short retention coverage.
    conversations = []
    for idx in range(8):
        conversations.append({"from": "user", "value": f"u{idx}", "audio": "audio/a.wav"})
        conversations.append({"from": "assistant", "value": f"a{idx}"})
    record = {"id": "conv-retain", "conversations": conversations}
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    output_dir = tmp_path / "out"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=output_dir,
        steps=2,
        save_every=1,
        keep_last_n_checkpoints=keep_last_n,
    )
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    result = _run_su_trainer(cfg_path)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    # Assert only the expected newest step folders remain on disk.
    checkpoint_dirs = _step_checkpoint_dirs(output_dir)
    assert [path.name for path in checkpoint_dirs] == expected_steps

    # Assert latest checkpoint metadata still reflects the terminal step.
    latest_state = _read_json(checkpoint_dirs[-1] / "trainer_state.json")
    assert latest_state["global_step"] == 2


def test_su_trainer_resolve_resume_dir_latest_and_explicit(tmp_path):
    """Resolve both latest and explicit checkpoint paths without launching training."""
    output_dir = tmp_path / "out"
    checkpoints = output_dir / "checkpoints"
    step1 = checkpoints / "step-00000001"
    step2 = checkpoints / "step-00000002"
    step1.mkdir(parents=True)
    step2.mkdir(parents=True)

    # `latest` should pick the highest step directory.
    assert _resolve_resume_dir(output_dir, "latest") == step2
    # Explicit path should return exactly what was provided.
    assert _resolve_resume_dir(output_dir, str(step1)) == step1


def test_su_trainer_resume_restores_training_state_equivalence(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Resuming from step K should reproduce the uninterrupted final weights at step N."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    # Use one repeated pair so each optimizer step sees the same batch semantics.
    record = {
        "id": "conv-equivalence",
        "conversations": [
            {"from": "user", "value": "u0", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a0"},
        ],
    }
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    # First run: uninterrupted reference trajectory to step 3.
    ref_output_dir = tmp_path / "out_ref"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    ref_cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=ref_output_dir,
        steps=3,
        # Keep only checkpoints needed by this test: resume step and final step.
        save_every=2,
        keep_last_n_checkpoints=2,
    )
    ref_cfg_path = tmp_path / "config_ref.yaml"
    _write_config(ref_cfg_path, ref_cfg)
    ref_result = _run_su_trainer(ref_cfg_path)
    ref_output = (ref_result.stdout or "") + (ref_result.stderr or "")
    assert ref_result.returncode == 0, ref_output

    # Second run: resume from step-2 checkpoint and finish at step 3.
    resumed_output_dir = tmp_path / "out_resumed"
    # Rebuild the same config for a different output directory.
    resumed_cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=resumed_output_dir,
        steps=3,
        # Resumed branch only needs its terminal checkpoint for weight comparison.
        keep_last_n_checkpoints=1,
    )
    resumed_cfg_path = tmp_path / "config_resumed.yaml"
    _write_config(resumed_cfg_path, resumed_cfg)
    step2_dir = ref_output_dir / "checkpoints" / "step-00000002"
    resumed_result = _run_su_trainer(
        resumed_cfg_path,
        extra_args=["--resume", str(step2_dir)],
    )
    resumed_output = (resumed_result.stdout or "") + (resumed_result.stderr or "")
    assert resumed_result.returncode == 0, resumed_output

    # Compare final checkpoint model weights between uninterrupted vs resumed training.
    ref_state = _load_model_state(ref_output_dir / "checkpoints" / "step-00000003")
    resumed_state = _load_model_state(resumed_output_dir / "checkpoints" / "step-00000003")
    assert ref_state.keys() == resumed_state.keys()
    for key in ref_state:
        assert torch.allclose(ref_state[key], resumed_state[key], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("compat_key", "saved_value", "current_value"),
    [
        ("training_steps", 2, 3),
        ("warmup_ratio", 0.03, 0.2),
        ("lr", 5.0e-5, 1.0e-4),
        ("max_grad_norm", 1.0, 0.5),
        ("batch_size", 1, 2),
        ("data_json_path", "/tmp/a.json", "/tmp/b.json"),
    ],
)
def test_su_trainer_validate_resume_compat_mismatch_keys(
    compat_key,
    saved_value,
    current_value,
):
    """Resume compatibility validator should fail fast on any mismatched key."""
    # Keep state minimal: only the key under test needs to differ.
    resume_state = {"compat": {compat_key: saved_value}}
    current_compat = {compat_key: current_value}

    with pytest.raises(ValueError, match=f"resume mismatch for {compat_key}"):
        _validate_resume_compat(resume_state, current_compat)


def test_su_trainer_validate_resume_compat_requires_metadata():
    """Resume compatibility validator should reject checkpoints missing compat metadata."""
    with pytest.raises(ValueError, match="missing compatibility metadata"):
        _validate_resume_compat({}, {"training_steps": 2})


def test_su_trainer_sharded_resume_cursor_is_conservative(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Resuming a sharded run should keep a conservative shard cursor."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    # Use a stable fixture shard directory so sharded resume tests stay deterministic.
    shard_dir = _fixtures_dir() / "su" / "two_shards"
    assert shard_dir.is_dir()

    output_dir = tmp_path / "out"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=shard_dir,
        audio_root=audio_root,
        output_dir=output_dir,
        # Keep training.steps fixed across resume compatibility checks.
        steps=3,
        # Keep step-2 resume source and step-3 terminal checkpoint.
        save_every=2,
        keep_last_n_checkpoints=2,
    )
    cfg.setdefault("data", {})
    cfg["data"]["shuffle_shards"] = True
    cfg["data"]["shuffle_seed"] = 123
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    # First run should end with cursor pointing at the last seen shard.
    first = _run_su_trainer(cfg_path)
    first_output = (first.stdout or "") + (first.stderr or "")
    assert first.returncode == 0, first_output
    step2_dir = output_dir / "checkpoints" / "step-00000002"
    step2_state = _read_json(step2_dir / "trainer_state.json")
    assert step2_state["next_shard_cursor"] == 1
    assert step2_state["shuffle_epoch"] == 0

    # Resume from step-2 should keep the conservative cursor.
    resumed = _run_su_trainer(cfg_path, extra_args=["--resume", str(step2_dir)])
    resumed_output = (resumed.stdout or "") + (resumed.stderr or "")
    assert resumed.returncode == 0, resumed_output

    # After one resumed step from the last shard, cursor stays conservative.
    step3_state = _read_json(output_dir / "checkpoints" / "step-00000003" / "trainer_state.json")
    assert step3_state["global_step"] == 3
    assert step3_state["next_shard_cursor"] == 1
    assert step3_state["shuffle_epoch"] == 0


def test_su_trainer_epoch_increments_on_exhaustion(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Trainer should advance shuffle_epoch after data exhaustion."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    shard_dir = _fixtures_dir() / "su" / "two_shards"
    assert shard_dir.is_dir()

    output_dir = tmp_path / "out"
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=shard_dir,
        audio_root=audio_root,
        output_dir=output_dir,
        # Force one more step than available batches to trigger exhaustion.
        steps=3,
        batch_size=1,
        grad_accum=1,
        num_workers=0,
    )
    cfg.setdefault("data", {})
    cfg["data"]["shuffle_shards"] = True
    cfg["data"]["shuffle_seed"] = 123
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    result = _run_su_trainer(cfg_path)
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    state = _read_json(output_dir / "checkpoints" / "step-00000003" / "trainer_state.json")
    assert state["shuffle_epoch"] == 1


def test_su_trainer_single_file_resume_wandb_and_cursor_contracts(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Single-file resume should enforce wandb policy and normalize shard cursor."""
    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    # Keep a minimal real single-file dataset for resume contract coverage.
    record = {
        "id": "conv-single-file-resume-contracts",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    output_dir = tmp_path / "out"
    # Reuse a shared config builder to avoid repeating boilerplate fields.
    cfg = _build_trainer_cfg(
        model_id=local_base_model_id,
        whisper_id=local_whisper_tiny_id,
        json_path=json_path,
        audio_root=audio_root,
        output_dir=output_dir,
        # Keep training.steps fixed across resume compatibility checks.
        steps=2,
        # Keep step-1 resume source and step-2 terminal checkpoint.
        save_every=1,
        keep_last_n_checkpoints=2,
    )
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    # Baseline checkpoint has no wandb run id because wandb is disabled.
    first = _run_su_trainer(cfg_path)
    first_output = (first.stdout or "") + (first.stderr or "")
    assert first.returncode == 0, first_output
    step1_dir = output_dir / "checkpoints" / "step-00000001"
    step1_state_path = step1_dir / "trainer_state.json"
    step1_state = _read_json(step1_state_path)
    assert step1_state["wandb_run_id"] is None

    # Enable wandb and verify strict resume fails on missing run id.
    cfg["wandb"] = {"enabled": True, "project": "pom-test"}
    _write_config(cfg_path, cfg)
    resumed_fail = _run_su_trainer(
        cfg_path,
        extra_args=["--resume", str(step1_dir)],
        extra_env={"WANDB_MODE": "offline"},
    )
    resumed_fail_output = (resumed_fail.stdout or "") + (resumed_fail.stderr or "")
    assert resumed_fail.returncode != 0
    assert "resume checkpoint is missing wandb_run_id" in resumed_fail_output

    # Inject a bogus cursor and validate single-file resume normalizes it to 0.
    step1_state["next_shard_cursor"] = 99
    step1_state_path.write_text(json.dumps(step1_state, indent=2), encoding="utf-8")

    # With override, resume should proceed and persist a new run id.
    resumed_ok = _run_su_trainer(
        cfg_path,
        extra_args=["--resume", str(step1_dir), "--allow-new-wandb-run-on-resume"],
        extra_env={"WANDB_MODE": "offline"},
    )
    resumed_ok_output = (resumed_ok.stdout or "") + (resumed_ok.stderr or "")
    assert resumed_ok.returncode == 0, resumed_ok_output
    step2_state = _read_json(output_dir / "checkpoints" / "step-00000002" / "trainer_state.json")
    assert step2_state["global_step"] == 2
    assert step2_state["next_shard_cursor"] == 0
    assert isinstance(step2_state["wandb_run_id"], str)
    assert step2_state["wandb_run_id"]
