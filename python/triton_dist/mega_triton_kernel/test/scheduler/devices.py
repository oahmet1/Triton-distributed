"""Device helpers for running the scheduler tests on non-CUDA hardware."""

from __future__ import annotations

from typing import Callable

from .torch_utils import torch


def _npu_available() -> bool:
    """Return ``True`` when PyTorch exposes a functional NPU backend."""

    return bool(
        hasattr(torch, "npu")
        and callable(getattr(torch.npu, "is_available", None))
        and torch.npu.is_available()
    )


def _patch_cuda_api(monkeypatch: "Callable[[object, str, object], None]", device: torch.device) -> None:
    """Monkeypatch the minimal CUDA interface used by the scheduler."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None, raising=False)


def resolve_scheduler_device(monkeypatch) -> torch.device:
    """Select the device that the scheduler tests should operate on."""

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())

    if _npu_available():
        device = torch.device("npu:0")
        _patch_cuda_api(monkeypatch, device)
        return device

    device = torch.device("cpu")
    _patch_cuda_api(monkeypatch, device)
    return device


__all__ = ["resolve_scheduler_device", "_npu_available"]
