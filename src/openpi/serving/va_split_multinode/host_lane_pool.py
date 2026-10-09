"""Host prefix lanes for cross-GPU VA-split.

POSIX shared memory stays on one machine. The TCP transport pickles the same
row dicts instead of attaching these slabs.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import time
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SlabSpec:
    key: str
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class PoolReady:
    slabs: tuple[SlabSpec, ...]
    max_lanes: int


def _register_host(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    try:
        import ctypes

        libcudart = ctypes.CDLL("libcudart.so")
        fn = libcudart.cudaHostRegister
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        fn.restype = ctypes.c_int
        ptr = ctypes.c_void_p(arr.ctypes.data)
        nbytes = ctypes.c_size_t(int(arr.nbytes))
        return int(fn(ptr, nbytes, 0)) == 0
    except (OSError, AttributeError):
        return False


class HostLanePool:
    """Dense per-lane numpy slabs. `ready[lane] == 1` means the row is readable."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        ready: np.ndarray,
        *,
        shms: list[shared_memory.SharedMemory] | None = None,
        owner: bool = False,
        host_registered: bool = False,
    ) -> None:
        if not arrays:
            raise ValueError("lane pool requires at least one array")
        lanes = int(ready.shape[0])
        for key, arr in arrays.items():
            if int(arr.shape[0]) != lanes:
                raise ValueError(f"{key} lane dim {arr.shape[0]} != {lanes}")
        self.arrays = arrays
        self.ready = ready
        self.max_lanes = lanes
        self.owner = owner
        self.host_registered = host_registered
        self._shms = list(shms or [])

    @classmethod
    def create(
        cls,
        example: dict[str, np.ndarray],
        *,
        max_lanes: int,
        shared: bool = False,
        try_host_register: bool = False,
    ) -> HostLanePool:
        if max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        if not example:
            raise ValueError("example row is empty")
        arrays: dict[str, np.ndarray] = {}
        shms: list[shared_memory.SharedMemory] = []
        registered = True
        for key, row in example.items():
            row_arr = np.asarray(row)
            shape = (int(max_lanes), *row_arr.shape)
            if shared:
                shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * row_arr.dtype.itemsize)
                shms.append(shm)
                arr = np.ndarray(shape, dtype=row_arr.dtype, buffer=shm.buf)
            else:
                arr = np.zeros(shape, dtype=row_arr.dtype)
            arr.fill(0)
            arrays[key] = arr
            if shared and try_host_register:
                registered = _register_host(arr) and registered
            elif not shared:
                registered = False
        if shared:
            ready_bytes = int(max_lanes) * np.dtype(np.int32).itemsize
            ready_shm = shared_memory.SharedMemory(create=True, size=ready_bytes)
            shms.append(ready_shm)
            ready = np.ndarray((int(max_lanes),), dtype=np.int32, buffer=ready_shm.buf)
        else:
            ready = np.zeros((int(max_lanes),), dtype=np.int32)
        ready.fill(0)
        return cls(arrays, ready, shms=shms, owner=shared, host_registered=registered and shared)

    def export_ready(self) -> PoolReady:
        if not self.owner:
            raise RuntimeError("only the owning pool can export shared-memory names")
        data_shms = self._shms[: len(self.arrays)]
        ready_shm = self._shms[len(self.arrays)]
        specs = [
            SlabSpec(key=key, name=shm.name, shape=tuple(int(x) for x in self.arrays[key].shape), dtype=self.arrays[key].dtype.str)
            for key, shm in zip(self.arrays, data_shms, strict=True)
        ]
        specs.append(
            SlabSpec(
                key="__ready__",
                name=ready_shm.name,
                shape=tuple(int(x) for x in self.ready.shape),
                dtype=self.ready.dtype.str,
            )
        )
        return PoolReady(slabs=tuple(specs), max_lanes=self.max_lanes)

    @classmethod
    def attach(cls, ready: PoolReady) -> HostLanePool:
        arrays: dict[str, np.ndarray] = {}
        shms: list[shared_memory.SharedMemory] = []
        ready_arr: np.ndarray | None = None
        for spec in ready.slabs:
            shm = shared_memory.SharedMemory(name=spec.name)
            shms.append(shm)
            arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
            if spec.key == "__ready__":
                ready_arr = arr
            else:
                arrays[spec.key] = arr
        if ready_arr is None:
            raise ValueError("shared pool is missing the ready slab")
        return cls(arrays, ready_arr, shms=shms, owner=False, host_registered=False)

    def row_template(self) -> dict[str, np.ndarray]:
        return {key: np.zeros(arr.shape[1:], dtype=arr.dtype) for key, arr in self.arrays.items()}

    def write_rows(self, lane_ids: list[int], rows: list[dict[str, np.ndarray]]) -> None:
        if len(lane_ids) != len(rows):
            raise ValueError("lane ids and rows must have the same length")
        for lane_id, row in zip(lane_ids, rows, strict=True):
            self._check_lane(lane_id)
            self.ready[lane_id] = 0
            for key, value in row.items():
                if key not in self.arrays:
                    raise KeyError(f"unexpected lane key {key}")
                np.copyto(self.arrays[key][lane_id], np.asarray(value), casting="no")
            self.ready[lane_id] = 1

    def read_row(self, lane_id: int) -> dict[str, np.ndarray]:
        self._check_lane(lane_id)
        return {key: np.array(arr[lane_id], copy=True) for key, arr in self.arrays.items()}

    def wait_ready(self, lane_id: int, *, timeout_s: float) -> None:
        self._check_lane(lane_id)
        deadline = time.perf_counter() + timeout_s
        while int(self.ready[lane_id]) == 0:
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"lane {lane_id} was not marked ready within {timeout_s}s")
            time.sleep(0.0005)

    def clear_ready(self, lane_id: int) -> None:
        self._check_lane(lane_id)
        self.ready[lane_id] = 0

    def close(self, *, unlink: bool = False) -> None:
        for shm in self._shms:
            shm.close()
            if unlink:
                shm.unlink()
        self._shms = []

    def _check_lane(self, lane_id: int) -> None:
        if lane_id < 0 or lane_id >= self.max_lanes:
            raise ValueError(f"lane {lane_id} outside 0..{self.max_lanes - 1}")


def unlink_pool(ready: PoolReady) -> None:
    for spec in ready.slabs:
        try:
            shm = shared_memory.SharedMemory(name=spec.name)
        except FileNotFoundError:
            continue
        shm.close()
        shm.unlink()


def stack_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("cannot stack an empty row list")
    keys = rows[0].keys()
    stacked: dict[str, np.ndarray] = {}
    for key in keys:
        stacked[key] = np.stack([np.asarray(row[key]) for row in rows], axis=0)
    return stacked


def slice_observation(observation: dict[str, Any], index: int) -> dict[str, Any]:
    def _slice(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _slice(item) for key, item in value.items()}
        arr = np.asarray(value)
        return arr[index]

    return _slice(observation)


def stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    def _stack(values: list[Any]) -> Any:
        first = values[0]
        if isinstance(first, dict):
            return {key: _stack([item[key] for item in values]) for key in first}
        return np.stack([np.asarray(item) for item in values], axis=0)

    return _stack(observations)
