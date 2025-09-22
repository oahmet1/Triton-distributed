"""Centralised Torch imports for the scheduler tests."""

from __future__ import annotations

import pytest

# ``pytest.importorskip`` ensures the test module is skipped when ``torch`` is
# unavailable instead of raising an ImportError during collection.
torch = pytest.importorskip("torch")

try:  # pragma: no cover - best-effort import for Ascend environments.
    import torch_npu  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]

__all__ = ["torch", "torch_npu"]
