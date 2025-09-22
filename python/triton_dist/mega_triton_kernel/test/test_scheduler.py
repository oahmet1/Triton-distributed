"""Smoke tests for the megakernel scheduler."""

from __future__ import annotations

import importlib
import sys
from typing import Iterable, List, Tuple

import pytest

from scheduler import (
    DummyDependency,
    DummyTask,
    DummyTaskFactory,
    LinearDependencyWorkload,
    SchedulerHarness,
    SchedulingStrategy,
    TaskTypeRegistry,
    enque_tasks,
)
from scheduler.torch_utils import torch


@pytest.fixture
def dummy_task_factory(scheduler_device: torch.device) -> DummyTaskFactory:
    return DummyTaskFactory(scheduler_device)


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
    dummy_task_factory: DummyTaskFactory,
    task_specs: List[Tuple[int, int, int, int]],
    expected_shape: Tuple[int, int, int],
) -> None:
    """Scoreboard dimensions grow with the maximum observed layer/task IDs and tile counts."""

    TaskTypeRegistry.reset_all_ids()
    tasks = [
        dummy_task_factory.create_task(
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


def test_megakernel_import_skips_heavy_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "triton_dist.mega_triton_kernel"

    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(module_name + "."):
            sys.modules.pop(key)

    real_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == module_name + ".tasks":
            raise AssertionError("tasks module should not be imported during package import")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
    assert "tasks" not in module.__dict__


def _expected_scoreboard_shape(tasks: Iterable[DummyTask]) -> Tuple[int, int, int]:
    max_layer_id = max(task.layer_id for task in tasks)
    max_task_id = max(task.task_id for task in tasks)
    max_tiles = max(task.num_tiles for task in tasks)
    return (
        max(max_layer_id, 1) + 1,
        max(max_task_id, 1) + 1,
        max(max_tiles, 1),
    )


def test_enque_tasks_round_robin(dummy_task_factory: DummyTaskFactory) -> None:
    """Ensure round-robin scheduling can enqueue dummy tasks."""

    TaskTypeRegistry.reset_all_ids()
    workload = LinearDependencyWorkload(dummy_task_factory)
    tasks = workload.build()
    harness = SchedulerHarness(num_sms=2, strategy=SchedulingStrategy.ROUND_ROBIN)

    expected_task_deps = harness.dependency_rows_after_opt(tasks)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ROUND_ROBIN,
        enable_dependency_opt=True,
    )

    assert num_tasks_tensor.cpu().tolist() == [2, 1]
    assert tuple(scoreboard.shape) == _expected_scoreboard_shape(tasks)
    assert task_deps_tensor.shape == (expected_task_deps, 2)

    wq_host = wq_tensor.cpu()
    task_type_id = DummyTask.get_task_type_id()
    uint32_max = torch.iinfo(torch.uint32).max

    assert int(wq_host[0, 0, 0].item()) == task_type_id
    assert int(wq_host[0, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 0, 0].item()) == task_type_id
    assert int(wq_host[1, 1, 0].item()) == uint32_max


def test_enque_tasks_zig_zag(dummy_task_factory: DummyTaskFactory) -> None:
    """Ensure zig-zag scheduling can enqueue dummy tasks."""

    TaskTypeRegistry.reset_all_ids()
    workload = LinearDependencyWorkload(dummy_task_factory)
    tasks = workload.build()
    harness = SchedulerHarness(num_sms=2, strategy=SchedulingStrategy.ZIG_ZAG)

    expected_task_deps = harness.dependency_rows_after_opt(tasks)
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=2,
        megakernel_tasks=tasks,
        strategy=SchedulingStrategy.ZIG_ZAG,
        enable_dependency_opt=True,
    )

    assert num_tasks_tensor.cpu().tolist() == [1, 2]
    assert tuple(scoreboard.shape) == _expected_scoreboard_shape(tasks)
    assert task_deps_tensor.shape == (expected_task_deps, 2)

    wq_host = wq_tensor.cpu()
    task_type_id = DummyTask.get_task_type_id()
    uint32_max = torch.iinfo(torch.uint32).max

    assert int(wq_host[0, 0, 0].item()) == task_type_id
    assert int(wq_host[0, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 1, 0].item()) == task_type_id
    assert int(wq_host[1, 0, 0].item()) == uint32_max


def test_scheduler_executes_dependent_addition_graph(
    dummy_task_factory: DummyTaskFactory,
) -> None:
    """Execute dependent addition tasks and validate dependency enforcement."""

    TaskTypeRegistry.reset_all_ids()
    harness = SchedulerHarness(num_sms=2, strategy=SchedulingStrategy.ROUND_ROBIN)
    torch.manual_seed(0)

    tensor_1 = torch.randn((4,), device=dummy_task_factory.device)
    tensor_2 = torch.randn((4,), device=dummy_task_factory.device)
    tensor_3 = torch.randn((4,), device=dummy_task_factory.device)
    tensor_4 = torch.randn((4,), device=dummy_task_factory.device)

    expected_first = tensor_1 + tensor_2
    expected_second = tensor_3 + tensor_4
    expected_final = expected_first + expected_second

    partial_first = torch.empty_like(expected_first)
    partial_second = torch.empty_like(expected_second)
    final_output = torch.empty_like(expected_final)

    executed_flags = {"first": False, "second": False}

    def _first_addition() -> None:
        executed_flags["first"] = True
        partial_first.copy_(tensor_1 + tensor_2)

    def _second_addition() -> None:
        executed_flags["second"] = True
        partial_second.copy_(tensor_3 + tensor_4)

    def _final_addition() -> None:
        assert executed_flags["first"], "final addition executed before first partial"
        assert executed_flags["second"], "final addition executed before second partial"
        torch.testing.assert_close(partial_first, expected_first)
        torch.testing.assert_close(partial_second, expected_second)
        final_output.copy_(partial_first + partial_second)

    first_task = dummy_task_factory.create_task(
        layer_id=0,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=1,
        dependency=[],
        io_tensors=[[tensor_1, tensor_2], [partial_first]],
        compute=_first_addition,
    )

    second_task = dummy_task_factory.create_task(
        layer_id=1,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=1,
        dependency=[],
        io_tensors=[[tensor_3, tensor_4], [partial_second]],
        compute=_second_addition,
    )

    final_task = dummy_task_factory.create_task(
        layer_id=2,
        task_id=0,
        tile_id_or_start=0,
        num_tiles=1,
        dependency=[
            DummyDependency(layer_id=0, task_id=0, start_tiles=0, end_tiles=1),
            DummyDependency(layer_id=1, task_id=0, start_tiles=0, end_tiles=1),
        ],
        io_tensors=[[partial_first, partial_second], [final_output]],
        compute=_final_addition,
    )

    tasks = [first_task, second_task, final_task]
    wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor = enque_tasks(
        num_sms=harness.num_sms,
        megakernel_tasks=tasks,
        strategy=harness.strategy,
        enable_dependency_opt=True,
    )

    assert num_tasks_tensor.cpu().tolist() == [2, 1]
    assert task_deps_tensor.shape == (2, 2)
    assert tuple(scoreboard.shape) == (3, 2, 1)

    execution_order = harness.execute(tasks)

    assert execution_order[-1] is final_task
    expected_task_keys = {
        (task.layer_id, task.task_id, task.tile_id_or_start) for task in tasks
    }
    observed_task_keys = {
        (task.layer_id, task.task_id, task.tile_id_or_start) for task in execution_order
    }
    assert observed_task_keys == expected_task_keys
    torch.testing.assert_close(partial_first, expected_first)
    torch.testing.assert_close(partial_second, expected_second)
    torch.testing.assert_close(final_output, expected_final)
    assert final_output.device == dummy_task_factory.device
