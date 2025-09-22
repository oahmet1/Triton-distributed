import importlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Type

import pytest

torch = pytest.importorskip("torch")

# Importing ``torch_npu`` (when available) registers the ``torch.npu`` namespace
# without requiring downstream code to guard optional imports.
try:  # pragma: no cover - the import is best-effort for Ascend environments.
    import torch_npu  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]


import triton_dist.mega_triton_kernel.core.scheduler as _scheduler
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


def _make_dummy_io_tensors(device: torch.device) -> List[List[torch.Tensor]]:
    """Allocate unique tensors for dummy task encoding."""

    input_tensor = torch.empty((4,), device=device, dtype=torch.float32)
    output_tensor = torch.empty((4,), device=device, dtype=torch.float32)
    return [[input_tensor], [output_tensor]]


def _make_dummy_task(
    device: torch.device,
    *,
    layer_id: int,
    task_id: int,
    tile_id_or_start: int,
    num_tiles: int,
    dependency: Iterable[DummyDependency] | None = None,
) -> DummyTask:
    return DummyTask(
        layer_id=layer_id,
        task_id=task_id,
        tile_id_or_start=tile_id_or_start,
        num_tiles=num_tiles,
        config=DummyConfig(),
        dependency=list(dependency) if dependency is not None else [],
        io_tensors=_make_dummy_io_tensors(device),
        extra_params={},
    )


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

    first_task = _make_dummy_task(
        device,
        layer_id=0,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=2,
    )

    second_task = _make_dummy_task(
        device,
        layer_id=1,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=3,
        dependency=[DummyDependency(layer_id=0, task_id=0, start_tiles=0, end_tiles=2)],
    )

    third_task = _make_dummy_task(
        device,
        layer_id=2,
        task_id=0,
        tile_id_or_start=1,
        num_tiles=1,
        dependency=[
            DummyDependency(layer_id=1, task_id=0, start_tiles=1, end_tiles=3),
            DummyDependency(layer_id=0, task_id=0, start_tiles=1, end_tiles=2),
        ],
    )

    return [first_task, second_task, third_task]


def _schedule_tasks_for_strategy(
    tasks: Iterable[DummyTask],
    num_sms: int,
    strategy: SchedulingStrategy,
) -> List[List[DummyTask]]:
    """Reproduce the scheduler's task-to-SM assignment for expectation building."""

    sm_wq_list: List[List[DummyTask]] = [[] for _ in range(num_sms)]
    if strategy == SchedulingStrategy.ROUND_ROBIN:
        for idx, task in enumerate(tasks):
            sm_wq_list[idx % num_sms].append(task)
        return sm_wq_list

    if strategy == SchedulingStrategy.ZIG_ZAG:
        zigzag_iter = 1
        for idx, task in enumerate(tasks):
            if idx % num_sms == 0:
                zigzag_iter ^= 1
            if zigzag_iter == 0:
                sm_wq_list[idx % num_sms].append(task)
            else:
                sm_wq_list[num_sms - 1 - (idx % num_sms)].append(task)
        return sm_wq_list

    raise NotImplementedError(f"unsupported strategy {strategy!r}")


def _count_dependencies_after_opt(
    tasks: Iterable[DummyTask],
    num_sms: int,
    strategy: SchedulingStrategy,
) -> int:
    """Return the expected dependency tensor rows after applying the optimiser."""

    scheduled_tasks = _schedule_tasks_for_strategy(tasks, num_sms=num_sms, strategy=strategy)

    expected_count = 0
    for queue in scheduled_tasks:
        deps_range: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for task in queue:
            for dep in task.dependency:
                key = dep.key()
                range_list = deps_range.get(key, [])
                covered = any(
                    left <= dep.start_tiles and right >= dep.end_tiles
                    for left, right in range_list
                )
                if not covered:
                    expected_count += 1
                deps_range[key] = range_list + [(dep.start_tiles, dep.end_tiles)]

    return expected_count


@pytest.mark.parametrize(
    ("task_specs", "expected_shape"),
    [
        pytest.param([(0, 0, 0, 1)], (2, 2, 1), id="single_task_minimum"),
        pytest.param([(3, 4, 0, 5)], (4, 5, 5), id="high_identifiers"),
        pytest.param(
            [(1, 0, 0, 2), (5, 7, 0, 1)],
            (6, 8, 2),
            id="multiple_tasks",
        ),
    ],
)
def test_work_queue_scoreboard_dimensions(
    scheduler_device: torch.device, task_specs: List[Tuple[int, int, int, int]], expected_shape
):
    """Scoreboard dimensions grow with the maximum observed layer/task IDs and tile counts."""

    TaskTypeRegistry.reset_all_ids()
    tasks = [
        _make_dummy_task(
            scheduler_device,
            layer_id=layer_id,
            task_id=task_id,
            tile_id_or_start=tile_start,
            num_tiles=num_tiles,
        )
        for (layer_id, task_id, tile_start, num_tiles) in task_specs
    ]

    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=1,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ROUND_ROBIN,
        enable_dependency_opt=True,
    )

    assert tuple(scoreboard.shape) == expected_shape
    assert num_tasks_tensor.cpu().tolist() == [len(task_specs)]
    assert task_deps_tensor.shape == (0, 2)
    assert wq_tensor.shape[0] == len(task_specs)
    assert wq_tensor.shape[1] == 1


def test_megakernel_import_skips_heavy_dependencies(monkeypatch):
    module_name = "triton_dist.mega_triton_kernel"

    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(module_name + "."):
            sys.modules.pop(key)

    real_import_module = importlib.import_module

    def guarded_import(name, package=None):
        if name == module_name + ".tasks":
            raise AssertionError("tasks module should not be imported during package import")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
    assert "tasks" not in module.__dict__


def test_enque_tasks_round_robin(scheduler_device: torch.device):
    """Ensure round-robin scheduling can enqueue dummy tasks."""

    TaskTypeRegistry.reset_all_ids()
    tasks = _build_dummy_tasks(scheduler_device)
    expected_task_deps = _count_dependencies_after_opt(
        tasks, num_sms=2, strategy=SchedulingStrategy.ROUND_ROBIN
    )
    max_layer_id = max(task.layer_id for task in tasks)
    max_task_id = max(task.task_id for task in tasks)
    max_tiles = max(task.num_tiles for task in tasks)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ROUND_ROBIN,
        enable_dependency_opt=True,
    )

    # sm0 receives tasks 0 and 2, sm1 receives task 1.
    assert num_tasks_tensor.cpu().tolist() == [2, 1]

    # Scoreboard dimensions reflect the maximum layer/task identifiers and tile count the
    # scheduler observed while materialising the work queues. The implementation clamps the
    # layer and task extents to at least two slots via the ``max_*`` seeds.
    expected_scoreboard_shape = (
        max(max_layer_id, 1) + 1,
        max(max_task_id, 1) + 1,
        max(max_tiles, 1),
    )
    assert tuple(scoreboard.shape) == expected_scoreboard_shape

    # Dependencies from the second and third task are encoded.
    assert task_deps_tensor.shape == (expected_task_deps, 2)

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
    expected_task_deps = _count_dependencies_after_opt(
        tasks, num_sms=2, strategy=SchedulingStrategy.ZIG_ZAG
    )
    max_layer_id = max(task.layer_id for task in tasks)
    max_task_id = max(task.task_id for task in tasks)
    max_tiles = max(task.num_tiles for task in tasks)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ZIG_ZAG,
        enable_dependency_opt=True,
    )

    # With zig-zag scheduling, sm1 receives two tasks and sm0 receives one.
    assert num_tasks_tensor.cpu().tolist() == [1, 2]

    # Shared invariants with the round-robin path.
    expected_scoreboard_shape = (
        max(max_layer_id, 1) + 1,
        max(max_task_id, 1) + 1,
        max(max_tiles, 1),
    )
    assert tuple(scoreboard.shape) == expected_scoreboard_shape
    assert task_deps_tensor.shape == (expected_task_deps, 2)

    wq_host = wq_tensor.cpu()
    task_type_id = DummyTask.get_task_type_id()
    uint32_max = torch.iinfo(torch.uint32).max

    assert int(wq_host[0, 0, 0].item()) == task_type_id
    assert int(wq_host[0, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 0, 0].item()) == uint32_max
