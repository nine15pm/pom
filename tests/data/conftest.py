import os

import pytest


@pytest.fixture(scope="session")
def local_base_model_id():
    """Return a local path when provided, otherwise fall back to HF repo id."""
    model_id = os.environ.get("POM_TEST_BASE_MODEL")
    if model_id:
        return model_id
    # Default to Hugging Face repo id and allow on-demand download/cache.
    return "Qwen/Qwen3-0.6B"
