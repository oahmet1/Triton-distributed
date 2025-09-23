"""Helpers that mirror the scheduler's behaviour for testing."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pytest

pytest.importorskip("torch")

from .dummy_task import DummyTask

import triton_dist.mega_triton_kernel.core.scheduler as _scheduler

SchedulingStrategy = _scheduler.SchedulingStrategy
enque_tasks = _scheduler.enque_tasks
_round_robin_scheduler = _scheduler.round_robin_scheduler
_zig_zag_scheduler = _scheduler.zig_zag_scheduler
_task_dependency_opt = _scheduler.task_dependency_opt


class SchedulerHarness:
    """Simulate the megakernel scheduler to aid in expectation building."""

    def __init__(self, num_sms: int, strategy: SchedulingStrategy) -> None:
        self.num_sms = num_sms
        self.strategy = strategy

    def schedule(self, tasks: Sequence[DummyTask]) -> List[List[DummyTask]]:
        """Assign tasks to SM work queues using the configured strategy."""

        task_list = list(tasks)
        if self.strategy == SchedulingStrategy.ROUND_ROBIN:
            sm_wq_list = _round_robin_scheduler(self.num_sms, task_list)
        elif self.strategy == SchedulingStrategy.ZIG_ZAG:
            sm_wq_list = _zig_zag_scheduler(self.num_sms, task_list)
        else:  # pragma: no cover - unreachable in current test coverage
            raise NotImplementedError(f"unsupported strategy {self.strategy!r}")

        return [list(queue) for queue in sm_wq_list]

    def dependency_rows_after_opt(self, tasks: Sequence[DummyTask]) -> int:
        """Mirror the dependency optimisation to derive tensor shapes."""

        scheduled_tasks = self.schedule(tasks)
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

    def execute(self, tasks: Sequence[DummyTask]) -> List[DummyTask]:
        """Execute the optimised work queues while respecting dependencies."""

        queues = _task_dependency_opt(self.schedule(tasks))
        return self._execute_sm_work_queues(queues)

    @staticmethod
    def _execute_sm_work_queues(sm_wq_list: Iterable[Iterable[DummyTask]]) -> List[DummyTask]:
        queues: List[List[DummyTask]] = [list(queue) for queue in sm_wq_list]
        completed_tiles: Set[Tuple[int, int, int]] = set()
        execution_order: List[DummyTask] = []

        while any(queues):
            progress = False
            for queue in queues:
                if not queue:
                    continue
                task = queue[0]
                if SchedulerHarness._dependencies_satisfied(task, completed_tiles):
                    queue.pop(0)
                    if task.compute is not None:
                        task.compute()
                    for tile_idx in range(
                        task.tile_id_or_start, task.tile_id_or_start + task.num_tiles
                    ):
                        completed_tiles.add((task.layer_id, task.task_id, tile_idx))
                    execution_order.append(task)
                    progress = True

            if not progress:
                pending = [
                    (
                        task.layer_id,
                        task.task_id,
                        [
                            (dep.layer_id, dep.task_id, dep.start_tiles, dep.end_tiles)
                            for dep in task.dependency
                        ],
                    )
                    for queue in queues
                    for task in queue
                ]
                raise RuntimeError(
                    "No executable tasks remain; unresolved dependencies: " f"{pending!r}"
                )

        return execution_order

    @staticmethod
    def _dependencies_satisfied(
        task: DummyTask, completed_tiles: Set[Tuple[int, int, int]]
    ) -> bool:
        for dep in task.dependency:
            for tile_idx in range(dep.start_tiles, dep.end_tiles):
                if (dep.layer_id, dep.task_id, tile_idx) not in completed_tiles:
                    return False
        return True


__all__ = ["SchedulerHarness", "SchedulingStrategy", "enque_tasks"]
