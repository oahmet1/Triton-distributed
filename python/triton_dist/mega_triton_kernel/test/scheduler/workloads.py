"""Reusable workloads for exercising the scheduler."""

from __future__ import annotations

from typing import List

from .dummy_task import DummyDependency, DummyTask, DummyTaskFactory


class LinearDependencyWorkload:
    """Construct a small chain of tasks with overlapping dependencies."""

    def __init__(self, factory: DummyTaskFactory) -> None:
        self.factory = factory

    def build(self) -> List[DummyTask]:
        first_task = self.factory.create_task(
            layer_id=0,
            task_id=0,
            tile_id_or_start=0,
            num_tiles=2,
        )

        second_task = self.factory.create_task(
            layer_id=1,
            task_id=0,
            tile_id_or_start=0,
            num_tiles=3,
            dependency=[DummyDependency(layer_id=0, task_id=0, start_tiles=0, end_tiles=2)],
        )

        third_task = self.factory.create_task(
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


__all__ = ["LinearDependencyWorkload"]
