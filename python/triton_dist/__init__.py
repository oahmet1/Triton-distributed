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
# yapf: disable
# forward import torch to load libtorch_cpu.so and libtorch_cuda.so
import torch  # noqa: F401
# yapf: enable

import sys
import types
import warnings

try:
    from . import language  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extension
    missing_triton = exc.name and exc.name.startswith("triton._C")
    if not missing_triton:
        raise

    warnings.warn(
        "triton_dist.language is unavailable because Triton distributed C++ extensions are not installed; "
        "functionality depending on them will be disabled.",
        RuntimeWarning,
    )

    language_stub = types.ModuleType("triton_dist.language")

    def _missing_attr(name: str):
        raise ModuleNotFoundError(
            "triton_dist.language requires Triton distributed C++ extensions, which are not installed",
        ) from exc

    language_stub.__getattr__ = _missing_attr  # type: ignore[attr-defined]
    sys.modules.setdefault(__name__ + ".language", language_stub)
    language = language_stub  # type: ignore[assignment]
