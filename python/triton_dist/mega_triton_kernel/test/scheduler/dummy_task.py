"""Dummy task implementations for the scheduler harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from .torch_utils import torch

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
    compute: Optional[Callable[[], None]] = None

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


class DummyTaskFactory:
    """Factory that creates dummy tasks targeting a specific device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def allocate_io(self) -> List[List[torch.Tensor]]:
        """Allocate unique tensors for dummy task encoding."""

        input_tensor = torch.empty((4,), device=self.device, dtype=torch.float32)
        output_tensor = torch.empty((4,), device=self.device, dtype=torch.float32)
        return [[input_tensor], [output_tensor]]

    def create_task(
        self,
        *,
        layer_id: int,
        task_id: int,
        tile_id_or_start: int,
        num_tiles: int,
        dependency: Iterable[DummyDependency] | None = None,
        io_tensors: Optional[List[List[torch.Tensor]]] = None,
        compute: Optional[Callable[[], None]] = None,
    ) -> DummyTask:
        return DummyTask(
            layer_id=layer_id,
            task_id=task_id,
            tile_id_or_start=tile_id_or_start,
            num_tiles=num_tiles,
            config=DummyConfig(),
            dependency=list(dependency) if dependency is not None else [],
            io_tensors=io_tensors if io_tensors is not None else self.allocate_io(),
            extra_params={},
            compute=compute,
        )


__all__ = [
    "DummyConfig",
    "DummyDependency",
    "DummyTask",
    "DummyTaskFactory",
    "TaskTypeRegistry",
    "MAX_NUM_TENSOR_DIMS",
]
