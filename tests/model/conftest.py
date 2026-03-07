import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    # Fail fast for model tests that must run on GPU.
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for model tests")
    return torch.device("cuda")
