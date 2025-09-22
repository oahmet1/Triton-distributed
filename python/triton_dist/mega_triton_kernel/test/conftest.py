"""PyTest fixtures shared across the scheduler tests."""

from __future__ import annotations

import pathlib
import sys

import pytest

_PROJECT_TEST_ROOT = pathlib.Path(__file__).resolve().parent
if str(_PROJECT_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_TEST_ROOT))

from scheduler.devices import resolve_scheduler_device
from scheduler.torch_utils import torch


@pytest.fixture
def scheduler_device(monkeypatch) -> torch.device:
    """Provide a device that can be consumed by the scheduler test suite."""

    return resolve_scheduler_device(monkeypatch)
