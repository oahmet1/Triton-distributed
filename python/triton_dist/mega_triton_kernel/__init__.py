################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Light-weight package initialization for the Triton megakernel stack.

Historically this module eagerly imported ``tasks`` and ``ModelBuilder`` to
provide backwards compatible attribute access. Those imports in turn pull in a
large portion of the Triton distributed runtime, including modules that touch
CUDA-specific entry points during module import. On environments without NVIDIA
GPUs (e.g. Huawei NPUs or CPU-only machines) those imports crash long before we
exercise the scheduler utilities that only rely on vanilla PyTorch tensors.

To keep the public API unchanged while avoiding the heavy side-effects we lazily
materialise the attributes on first access using ``__getattr__``. This allows
callers that only need the scheduler to import
``triton_dist.mega_triton_kernel`` without triggering CUDA initialisation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ("ModelBuilder", "tasks")

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from . import tasks as _tasks
    from .models.model_builder import ModelBuilder as _ModelBuilder


def __getattr__(name: str) -> Any:
    if name == "tasks":
        module = import_module(f"{__name__}.tasks")
        globals()[name] = module
        return module
    if name == "ModelBuilder":
        from .models.model_builder import ModelBuilder as model_builder

        globals()[name] = model_builder
        return model_builder
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
