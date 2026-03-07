import os
from pathlib import Path

import pytest
import torchaudio


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AUDIO_DIR = FIXTURES_DIR / "audio"


@pytest.fixture(scope="session")
def fixture_audio_paths():
    # Collect shared WAV fixtures for audio-based tests.
    if not AUDIO_DIR.exists():
        pytest.fail(f"Missing audio fixtures at {AUDIO_DIR}")
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if len(wavs) < 2:
        pytest.fail("Expected at least 2 WAV fixtures")
    return wavs


@pytest.fixture(scope="session")
def load_fixture_audio():
    # Provide a reusable loader for WAV fixtures.
    def _load(path: Path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    return _load


@pytest.fixture(scope="session")
def base_model_id():
    return os.environ.get("POM_TEST_BASE_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def whisper_tiny_id():
    return os.environ.get("POM_TEST_WHISPER_TINY", "openai/whisper-tiny")
