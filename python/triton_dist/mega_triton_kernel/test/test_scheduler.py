import pytest
from dataclasses import dataclass

torch = pytest.importorskip("torch")

# Importing ``torch_npu`` (when available) registers the ``torch.npu`` namespace
# without requiring downstream code to guard optional imports.
try:  # pragma: no cover - the import is best-effort for Ascend environments.
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]

from triton_dist.mega_triton_kernel.core.config import ConfigBase
from triton_dist.mega_triton_kernel.core.scheduler import SchedulingStrategy, enque_tasks
from triton_dist.mega_triton_kernel.core.task_base import (
    TaskBase,
    TaskDependency,
    TaskIDManager,
)


@dataclass
class DummyConfig(ConfigBase):
    """Trivial configuration for dummy tasks."""


@dataclass
class DummyTask(TaskBase):
    """Minimal ``TaskBase`` implementation for exercising the scheduler."""
    config: DummyConfig


def _npu_available() -> bool:
    """Return ``True`` when PyTorch exposes a functional NPU backend."""

    return bool(
        hasattr(torch, "npu")
        and callable(getattr(torch.npu, "is_available", None))
        and torch.npu.is_available()
    )


@pytest.fixture
def scheduler_device(monkeypatch) -> torch.device:
    """Provide a device that can be consumed by the scheduler test suite.

    The production scheduler currently hard-codes CUDA tensors. For test
    coverage on systems without NVIDIA GPUs (e.g., Ascend NPUs or CPU-only
    environments), we monkeypatch the minimal subset of ``torch.cuda`` APIs
    that the scheduler relies on so the code executes against the best
    available accelerator or falls back to CPU tensors.
    """

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())

    if _npu_available():
        device = torch.device("npu:0")

        monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None, raising=False)
        return device

    device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None, raising=False)
    return device


def _build_dummy_tasks(device: torch.device):
    """Construct a small set of tasks with simple data dependencies."""

    def _make_io_tensors():
        # Create unique tensors to ensure distinct data pointers for encoding.
        input_tensor = torch.empty((4,), device=device, dtype=torch.float32)
        output_tensor = torch.empty((4,), device=device, dtype=torch.float32)
        return [[input_tensor], [output_tensor]]

    first_task = DummyTask(
        layer_id=0,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=2,
        config=DummyConfig(),
        dependency=[],
        io_tensors=_make_io_tensors(),
        extra_params={},
    )

    second_task = DummyTask(
        layer_id=1,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=3,
        config=DummyConfig(),
        dependency=[TaskDependency(layer_id=0, task_id=0, start_tiles=0, end_tiles=2)],
        io_tensors=_make_io_tensors(),
        extra_params={},
    )

    third_task = DummyTask(
        layer_id=2,
        task_id=0,
        tile_id_or_start=1,
        num_tiles=1,
        config=DummyConfig(),
        dependency=[
            TaskDependency(layer_id=1, task_id=0, start_tiles=1, end_tiles=3),
            TaskDependency(layer_id=0, task_id=0, start_tiles=1, end_tiles=2),
        ],
        io_tensors=_make_io_tensors(),
        extra_params={},
    )

    return [first_task, second_task, third_task]


def test_enque_tasks_round_robin(scheduler_device: torch.device):
    """Ensure round-robin scheduling can enqueue dummy tasks."""

    TaskIDManager.reset_all_ids()
    tasks = _build_dummy_tasks(scheduler_device)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ROUND_ROBIN,
        enable_dependency_opt=True,
    )

    # sm0 receives tasks 0 and 2, sm1 receives task 1.
    assert num_tasks_tensor.cpu().tolist() == [2, 1]

    # Scoreboard should track three layers, one task per layer, and the maximum tile count (3).
    assert scoreboard.shape == (3, 1, 3)

    # Dependencies from the second and third task are encoded.
    assert task_deps_tensor.shape == (3, 2)

    # Verify that valid task entries share the same dummy type id and padding is set to uint32 max.
    wq_host = wq_tensor.cpu()
    task_type_id = DummyTask.get_task_type_id()
    uint32_max = torch.iinfo(torch.uint32).max

    assert int(wq_host[0, 0, 0].item()) == task_type_id
    assert int(wq_host[0, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 0, 0].item()) == task_type_id
    assert int(wq_host[1, 1, 0].item()) == uint32_max


def test_enque_tasks_zig_zag(scheduler_device: torch.device):
    """Ensure zig-zag scheduling can enqueue dummy tasks."""

    TaskIDManager.reset_all_ids()
    tasks = _build_dummy_tasks(scheduler_device)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ZIG_ZAG,
        enable_dependency_opt=True,
    )

    # With zig-zag scheduling, sm1 receives two tasks and sm0 receives one.
    assert num_tasks_tensor.cpu().tolist() == [1, 2]

    # Shared invariants with the round-robin path.
    assert scoreboard.shape == (3, 1, 3)
    assert task_deps_tensor.shape == (3, 2)

    wq_host = wq_tensor.cpu()
    task_type_id = DummyTask.get_task_type_id()
    uint32_max = torch.iinfo(torch.uint32).max

    assert int(wq_host[0, 0, 0].item()) == task_type_id
    assert int(wq_host[0, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 0, 0].item()) == uint32_max
