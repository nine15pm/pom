import os
from pathlib import Path
import shutil

import pytest
import torch

from inference.loader import load_runtime

_DEFAULT_POM_CACHE_DIR = Path("~/.cache/pom").expanduser()


def _build_trained_inference_cfg(*, cache_dir: str) -> dict:
    """Build one minimal inference config that points at a trained artifact cache."""
    # Keep test decode settings aligned with real inference defaults.
    return {
        "models": {
            "artifact_mode": "hf",
            "cache_dir": cache_dir,
            "base_cache_dir": None,
            "speech_encoder_cache_dir": None,
            "speech_vocab_size": 6561,
            "checkpoint": None,
        },
        "tokenizer": {
            "source": None,
            "enable_thinking": False,
            "assistant_stop_token": None,
        },
        "generation": {
            "text": {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9},
            "speech": {
                "max_new_tokens": 256,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "max_repeat_run": 30,
                "read_length": 3,
                "write_length": 10,
            },
        },
        "runtime": {"device": "cuda", "dtype": "bf16", "seed": 42},
        "output": {"output_wav_path": None},
    }


@pytest.fixture(scope="session", autouse=True)
def require_cuda_for_inference_tests():
    # Inference runtime contract is CUDA-only.
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for inference tests")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def decoder_dir(trained_cache_dir: str) -> str:
    """Resolve decoder assets path from env override or default trained cache layout."""
    env_value = os.environ.get("POM_TEST_DECODER_DIR")
    if env_value is not None:
        path = Path(env_value).expanduser()
    else:
        # Default to the canonical cache layout: <cache_dir>/decoder.
        path = Path(trained_cache_dir) / "decoder"
    if not path.is_dir():
        pytest.fail(
            f"decoder assets directory not found: {path} "
            "(set POM_TEST_DECODER_DIR to override)"
        )
    return path.as_posix()


@pytest.fixture(scope="session")
def trained_cache_dir() -> str:
    """Resolve trained artifact cache dir from env override or ~/.cache/pom default."""
    env_value = os.environ.get("POM_TEST_TRAINED_CACHE_DIR")
    if env_value is not None:
        root = Path(env_value).expanduser()
    else:
        # Match the default cache root documented in configs/inference.yaml.
        root = _DEFAULT_POM_CACHE_DIR
    if not root.is_dir():
        pytest.fail(
            f"trained cache directory not found: {root} "
            "(set POM_TEST_TRAINED_CACHE_DIR to override)"
        )
    for required in ("thinker", "talker", "tokenizer"):
        if not (root / required).is_dir():
            pytest.fail(f"trained cache is missing required subdir: {required} under {root}")
    return root.as_posix()


@pytest.fixture(scope="session")
def trained_runtime_and_cfg(trained_cache_dir: str):
    """Load one real runtime bundle from trained artifacts for inference/streaming tests."""
    cfg = _build_trained_inference_cfg(cache_dir=trained_cache_dir)
    runtime = load_runtime(cfg)
    return runtime, cfg


def _remove_path_if_exists(path: Path) -> None:
    """Remove one file/symlink/directory path when present."""
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


@pytest.fixture(scope="function")
def trained_runtime_no_decoder_and_cfg(trained_cache_dir: str, tmp_path):
    """Load one real trained runtime copy with decoder assets removed."""
    cache_copy = tmp_path / "trained_cache_no_decoder"
    shutil.copytree(Path(trained_cache_dir), cache_copy)
    _remove_path_if_exists(cache_copy / "decoder")

    cfg = _build_trained_inference_cfg(cache_dir=cache_copy.as_posix())
    runtime = load_runtime(cfg)
    if getattr(runtime, "decoder", None) is not None:
        pytest.fail("expected runtime without decoder assets")
    return runtime, cfg


@pytest.fixture(scope="function")
def trained_runtime_with_decoder_and_cfg(
    trained_cache_dir: str,
    decoder_dir: str,
    tmp_path,
):
    """Load one real trained runtime copy with decoder assets present."""
    cache_copy = tmp_path / "trained_cache_with_decoder"
    shutil.copytree(Path(trained_cache_dir), cache_copy)
    decoder_target = cache_copy / "decoder"
    _remove_path_if_exists(decoder_target)
    decoder_target.symlink_to(Path(decoder_dir), target_is_directory=True)

    cfg = _build_trained_inference_cfg(cache_dir=cache_copy.as_posix())
    runtime = load_runtime(cfg)
    if getattr(runtime, "decoder", None) is None:
        pytest.fail("expected runtime with decoder assets")
    return runtime, cfg
