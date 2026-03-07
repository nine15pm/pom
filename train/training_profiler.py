"""Shared per-step profiling helpers for SU and TTS trainers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator


def _find_batch_size(value: Any) -> Optional[int]:
    """Find the leading batch dimension from nested batch structures."""
    if torch.is_tensor(value):
        return int(value.shape[0])
    if isinstance(value, dict):
        # Walk nested mappings and return the first tensor batch size.
        for item in value.values():
            size = _find_batch_size(item)
            if size is not None:
                return size
    if isinstance(value, (list, tuple)):
        # Walk nested sequences and return the first tensor batch size.
        for item in value:
            size = _find_batch_size(item)
            if size is not None:
                return size
    return None


def infer_micro_batch_size(batch: Any) -> int:
    """Infer micro-batch size from the first tensor in a batch."""
    size = _find_batch_size(batch)
    if size is None:
        raise ValueError("unable to infer micro-batch size from batch")
    return size


class StepProfiler:
    """Track timing/throughput/memory stats for one optimizer step window."""

    def __init__(
        self,
        *,
        enabled: bool,
        use_cuda: bool,
        device: torch.device,
        world_size: int,
        grad_accum: int,
        effective_target: Optional[int],
    ) -> None:
        """Store static config and initialize per-window state."""
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.device = device
        self.world_size = world_size
        self.grad_accum = grad_accum
        self.effective_target = effective_target
        self._new_window = True
        self._step_t0 = 0.0
        self._data_time_total = 0.0
        self._local_samples = 0
        self._micro_batch: Optional[int] = None

    def start_window(self) -> None:
        """Start a new accumulation window for step-level profiling."""
        if not self.enabled or not self._new_window:
            return
        if self.use_cuda:
            # Scope peak memory stats to this optimizer step window.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # Start wall-clock timing for the full optimizer step window.
        self._step_t0 = time.perf_counter()
        self._data_time_total = 0.0
        self._local_samples = 0
        self._micro_batch = None
        self._new_window = False

    def record_batch(self, batch: Any, data_time: float) -> None:
        """Accumulate dataloader wait time and sample count for this window."""
        if not self.enabled:
            return
        micro = infer_micro_batch_size(batch)
        if self._micro_batch is None:
            self._micro_batch = micro
        self._local_samples += micro
        self._data_time_total += data_time

    def finalize_step(self, accelerator: Accelerator) -> Dict[str, Any]:
        """Finalize and return profile metrics at optimizer-step boundaries."""
        if not self.enabled:
            self._new_window = True
            return {}
        # Stop timing after queued CUDA work for the window is complete.
        if self.use_cuda:
            torch.cuda.synchronize()
        step_time = time.perf_counter() - self._step_t0
        if step_time <= 0:
            step_time = 1e-6

        # Reduce local sample counts to global sample count for throughput.
        samples_tensor = torch.tensor(float(self._local_samples), device=self.device)
        global_samples = accelerator.reduce(samples_tensor, reduction="sum")
        samples_per_sec = global_samples.item() / step_time

        if self.use_cuda:
            # Gather per-rank peaks and report worst-case GPU memory.
            peak_alloc = float(torch.cuda.max_memory_allocated())
            peak_reserved = float(torch.cuda.max_memory_reserved())
            mem_stats = torch.tensor([peak_alloc, peak_reserved], device=self.device)
            mem_stats_all = accelerator.gather(mem_stats)
            if mem_stats_all.ndim == 1:
                mem_stats_all = mem_stats_all.unsqueeze(0)
            peak_alloc_mib = mem_stats_all[:, 0].max().item() / (1024**2)
            peak_reserved_mib = mem_stats_all[:, 1].max().item() / (1024**2)
        else:
            peak_alloc_mib = 0.0
            peak_reserved_mib = 0.0

        # Keep both configured and observed effective batch views in logs.
        micro_batch = self._micro_batch or 0
        effective_target = (
            self.effective_target
            if self.effective_target is not None
            else micro_batch * self.world_size * self.grad_accum
        )
        effective_samples = int(global_samples.item())

        self._new_window = True
        return {
            "perf/step_time_sec": step_time,
            "perf/data_time_sec": self._data_time_total,
            "perf/samples_per_sec": samples_per_sec,
            "gpu/peak_mem_alloc_mib": peak_alloc_mib,
            "gpu/peak_mem_reserved_mib": peak_reserved_mib,
            "batch/micro": micro_batch,
            "batch/grad_accum": self.grad_accum,
            "batch/world_size": self.world_size,
            "batch/effective_target": effective_target,
            "batch/effective_samples": effective_samples,
        }
