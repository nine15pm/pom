import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    """Fail fast if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for trainer tests")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def local_base_model_id():
    """Return a local path when provided, otherwise fall back to HF repo id."""
    model_id = os.environ.get("POM_TEST_BASE_MODEL")
    if model_id:
        return model_id
    # Default to Hugging Face repo id and allow on-demand download/cache.
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def local_whisper_tiny_id():
    """Return a local path when provided, otherwise fall back to HF repo id."""
    model_id = os.environ.get("POM_TEST_WHISPER_TINY")
    if model_id:
        return model_id
    # Default to Hugging Face repo id and allow on-demand download/cache.
    return "openai/whisper-tiny"
