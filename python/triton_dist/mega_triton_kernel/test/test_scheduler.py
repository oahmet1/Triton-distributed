import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type

import pytest

torch = pytest.importorskip("torch")

# Importing ``torch_npu`` (when available) registers the ``torch.npu`` namespace
# without requiring downstream code to guard optional imports.
try:  # pragma: no cover - the import is best-effort for Ascend environments.
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]


def _load_scheduler_module():
    """Load the scheduler directly from its source file without package side-effects."""

    module_name = "_triton_dist_scheduler"
    scheduler_path = Path(__file__).resolve().parent.parent / "core" / "scheduler.py"
    spec = importlib.util.spec_from_file_location(module_name, scheduler_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_scheduler = _load_scheduler_module()
SchedulingStrategy = _scheduler.SchedulingStrategy
enque_tasks = _scheduler.enque_tasks


MAX_NUM_TENSOR_DIMS = 4


class TaskTypeRegistry:
    """Simple stand-in for the production ``TaskIDManager``."""

    _type_id_counter: int = 0
    _type_id_map: Dict[Type[object], int] = {}

    @classmethod
    def get_task_type_id(cls, task_cls: Type[object]) -> int:
        if task_cls not in cls._type_id_map:
            cls._type_id_map[task_cls] = cls._type_id_counter
            cls._type_id_counter += 1
        return cls._type_id_map[task_cls]

    @classmethod
    def reset_all_ids(cls) -> None:
        cls._type_id_counter = 0
        cls._type_id_map.clear()


@dataclass
class DummyConfig:
    """Trivial configuration for dummy tasks."""


@dataclass
class DummyDependency:
    layer_id: int
    task_id: int
    start_tiles: int
    end_tiles: int

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.task_id)


@dataclass
class DummyTask:
    """Minimal task implementation for exercising the scheduler."""

    layer_id: int
    task_id: int
    tile_id_or_start: int
    num_tiles: int
    config: DummyConfig
    dependency: List[DummyDependency]
    io_tensors: List[List[torch.Tensor]]
    extra_params: Dict[str, int]

    @classmethod
    def get_task_type_id(cls) -> int:
        return TaskTypeRegistry.get_task_type_id(cls)

    def _io_to_tuple(self) -> Tuple[int, ...]:
        tensors: Iterable[torch.Tensor] = self.io_tensors[0] + self.io_tensors[1]
        entries: List[int] = []
        for tensor in tensors:
            data_ptr = tensor.data_ptr()
            ptr_low = data_ptr & 0xFFFFFFFF
            ptr_high = (data_ptr >> 32) & 0xFFFFFFFF

            shape = list(tensor.shape)
            if len(shape) > MAX_NUM_TENSOR_DIMS:
                raise ValueError("unexpected tensor rank in dummy task")
            padded_shape = shape + [1] * (MAX_NUM_TENSOR_DIMS - len(shape))

            tensor_fields = [ptr_low, ptr_high, *padded_shape]
            if len(tensor_fields) % 2 != 0:
                raise AssertionError("tensor metadata must maintain 64-bit alignment")
            entries.extend(tensor_fields)

        return tuple(entries)

    def _extra_params_to_tuple(self) -> Tuple[int, ...]:
        assert not self.extra_params, "dummy tasks do not support extra params"
        return ()

    def encoding_with_deps(self, deps_l: int, deps_r: int) -> Tuple[int, ...]:
        entries: List[int] = [
            self.get_task_type_id(),
            self.layer_id,
            self.task_id,
            self.tile_id_or_start,
            deps_l,
            deps_r,
        ]

        io_tuple = self._io_to_tuple()
        if len(entries) % 2 != 0:
            raise AssertionError("task header must maintain 64-bit alignment")
        entries.extend(io_tuple)
        entries.extend(self._extra_params_to_tuple())

        for value in entries:
            if not isinstance(value, int):
                raise TypeError(f"expected int in encoding, got {type(value)!r}")
        return tuple(entries)


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
        dependency=[DummyDependency(layer_id=0, task_id=0, start_tiles=0, end_tiles=2)],
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
            DummyDependency(layer_id=1, task_id=0, start_tiles=1, end_tiles=3),
            DummyDependency(layer_id=0, task_id=0, start_tiles=1, end_tiles=2),
        ],
        io_tensors=_make_io_tensors(),
        extra_params={},
    )

    return [first_task, second_task, third_task]


def test_enque_tasks_round_robin(scheduler_device: torch.device):
    """Ensure round-robin scheduling can enqueue dummy tasks."""

    TaskTypeRegistry.reset_all_ids()
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

    TaskTypeRegistry.reset_all_ids()
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
