"""Tiny Python API entrypoint for offline inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from inference.config import InferenceConfig, load_inference_config
from inference.loader import RuntimeBundle, load_runtime
from inference.offline import InferenceRequest, InferenceResponse, run_turn as run_pipeline_turn
from dataclasses import dataclass


@dataclass
class PomInferenceAPI:
    """Own one loaded runtime and expose one clean offline call."""

    cfg: InferenceConfig
    runtime: RuntimeBundle

    @classmethod
    def from_config(
        cls,
        path: str | Path = "configs/inference.yaml",
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> "PomInferenceAPI":
        """Build API from validated config and one startup runtime load."""
        cfg = load_inference_config(path=path, overrides=overrides)
        runtime = load_runtime(cfg)
        return cls(cfg=cfg, runtime=runtime)

    def run_turn(
        self,
        *,
        audio_path: str,
        decode_audio: bool = True,
        text_generation_overrides: dict[str, object] | None = None,
        speech_generation_overrides: dict[str, object] | None = None,
    ) -> InferenceResponse:
        """Run one turn through the shared core inference pipeline."""
        request = InferenceRequest(
            audio_path=audio_path,
            decode_audio=decode_audio,
            text_generation_overrides=text_generation_overrides,
            speech_generation_overrides=speech_generation_overrides,
        )
        return run_pipeline_turn(runtime=self.runtime, cfg=self.cfg, request=request)


__all__ = ["PomInferenceAPI"]
