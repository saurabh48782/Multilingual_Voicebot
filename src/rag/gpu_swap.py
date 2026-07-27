"""Page RAG model weights onto the GPU only for the duration of a call.

The e5 embedder and the bge cross-encoder together hold ~5 GB of VRAM, but
they are only used during `retrieve`, which finishes long before `synthesize`
pages Indic Parler-TTS in.

Same pattern the STT/TTS sidecars use (`services/*/_GpuSwap`): keep the
weights page-locked in CPU RAM, copy them to the GPU inside `on_gpu()`, and
hand the VRAM straight back to the driver afterwards. No-op on CPU-only hosts.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


def resolve_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class GpuSwap:
    """Hold `model`'s weights pinned in CPU RAM, paging them onto `device`
    only for the span of an `on_gpu()` block.

    `on_gpu()` is serialised by an internal lock: two concurrent retrievals
    must not page the same model in and out of the GPU at the same time.
    Re-entrant use from the same thread is not supported.
    """

    def __init__(self, model: nn.Module, device: str) -> None:
        self._model = model
        self._device = device
        self._enabled = torch.cuda.is_available() and str(device) != "cpu"
        self._lock = threading.Lock()
        self._cpu_params: dict[str, torch.Tensor] = {}
        self._cpu_buffers: dict[str, torch.Tensor] = {}
        if self._enabled:
            # named_parameters()/named_buffers() dedupe shared storage, so
            # tied weights are pinned and restored once and stay tied.
            for name, p in model.named_parameters():
                p.data = p.data.pin_memory()
                self._cpu_params[name] = p.data
            for name, b in model.named_buffers():
                if b.device.type == "cpu":
                    b.data = b.data.pin_memory()
                    self._cpu_buffers[name] = b.data

    @contextmanager
    def on_gpu(self) -> Iterator[None]:
        with self._lock:
            if not self._enabled:
                yield
                return
            self._model.to(self._device, non_blocking=True)
            try:
                yield
            finally:
                for name, p in self._model.named_parameters():
                    p.data = self._cpu_params[name]
                for name, b in self._model.named_buffers():
                    cpu = self._cpu_buffers.get(name)
                    if cpu is not None:
                        b.data = cpu
                torch.cuda.empty_cache()
