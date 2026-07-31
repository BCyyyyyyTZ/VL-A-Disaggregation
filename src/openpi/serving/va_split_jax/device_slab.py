from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
import os
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv import drvapi


@dataclass(frozen=True, slots=True)
class DeviceSlabSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    max_lanes: int

    @property
    def slab_shape(self) -> tuple[int, ...]:
        return (self.max_lanes, *self.shape[1:])


@dataclass(frozen=True, slots=True)
class DeviceSlabHandle:
    spec: DeviceSlabSpec
    transport: str
    device_ordinal: int
    handle_bytes: bytes
    ready_event_bytes: bytes | None = None
    offset: int = 0
    strides: tuple[int, ...] | None = None


class DeviceSlab:
    def __init__(
        self,
        spec: DeviceSlabSpec,
        array: jax.Array,
        handle: DeviceSlabHandle,
        close_stack: contextlib.ExitStack | None = None,
    ):
        self.spec = spec
        self.array = array
        self.handle = handle
        self._close_stack = close_stack

    def close(self) -> None:
        if self._close_stack is not None:
            self._close_stack.close()
            self._close_stack = None

    def __enter__(self) -> DeviceSlab:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class DeviceSlabBackend:
    transport = "device-ipc"

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        """Allocate a process-shareable device slab described by spec."""
        raise NotImplementedError

    def open_slab(self, handle: DeviceSlabHandle) -> DeviceSlab:
        """Open a producer-created device slab from another process."""
        raise NotImplementedError

    def copy_lane_from_array(self, slab: DeviceSlab, lane_id: int, value: jax.Array) -> DeviceSlab:
        _validate_lane_update(slab, lane_id, value)
        update = jax.lax.dynamic_update_slice_in_dim(slab.array, value, lane_id, axis=0)
        update.block_until_ready()
        return DeviceSlab(slab.spec, update, slab.handle, slab._close_stack)

    def view_batch(self, slab: DeviceSlab, batch_size: int) -> jax.Array:
        if batch_size < 0 or batch_size > slab.spec.max_lanes:
            raise ValueError(f"batch_size {batch_size} outside slab capacity {slab.spec.max_lanes}")
        return jax.lax.dynamic_slice_in_dim(slab.array, 0, batch_size, axis=0)

    def slice_lanes(self, slab: DeviceSlab, slot_ids: tuple[int, ...]) -> jax.Array:
        """Return a dense-prefix device-side view for already-mapped lanes without reopening IPC."""
        if slot_ids != tuple(range(len(slot_ids))):
            raise ValueError("The first implementation only permits dense-prefix slot ids")
        return self.view_batch(slab, len(slot_ids))


class CudaIpcDeviceSlabBackend(DeviceSlabBackend):
    transport = "cuda-ipc-numba"

    def __init__(self, *, device_ordinal: int | None = None):
        self._device_ordinal = device_ordinal

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        if spec.max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        dtype = np.dtype(spec.dtype)
        if self._device_ordinal is not None:
            device = jax.devices("gpu")[self._device_ordinal]
            array = jax.device_put(jnp.zeros(spec.slab_shape, dtype=dtype), device)
        else:
            array = jnp.zeros(spec.slab_shape, dtype=dtype)
        array.block_until_ready()
        if not _is_gpu_array(array):
            raise RuntimeError("CudaIpcDeviceSlabBackend requires a GPU-backed JAX array")
        handle = _export_cuda_ipc_handle(spec, array)
        return DeviceSlab(spec, array, handle)

    def open_slab(self, handle: DeviceSlabHandle) -> DeviceSlab:
        if handle.transport != self.transport:
            raise RuntimeError(f"unsupported slab transport {handle.transport!r}")
        cuda.select_device(handle.device_ordinal)
        stack = contextlib.ExitStack()
        try:
            ipc_array = stack.enter_context(
                cuda.open_ipc_array(
                    tuple(handle.handle_bytes),
                    handle.spec.slab_shape,
                    np.dtype(handle.spec.dtype),
                    strides=handle.strides,
                    offset=handle.offset,
                )
            )
            array = jnp.asarray(ipc_array)
            array.block_until_ready()
            if array.unsafe_buffer_pointer() != ipc_array.device_ctypes_pointer.value:
                raise RuntimeError("JAX imported IPC slab by copy instead of aliasing the opened device allocation")
            return DeviceSlab(handle.spec, array, handle, stack)
        except Exception:
            stack.close()
            raise

    def copy_lane_from_array(self, slab: DeviceSlab, lane_id: int, value: jax.Array) -> DeviceSlab:
        _validate_lane_update(slab, lane_id, value)
        if slab.handle.transport != self.transport:
            raise RuntimeError(f"expected {self.transport!r} slab, got {slab.handle.transport!r}")
        value.block_until_ready()
        slab.array.block_until_ready()
        cuda.select_device(slab.handle.device_ordinal)
        dst = cuda.from_cuda_array_interface(_cuda_array_interface_for_lane(slab.array, slab.spec, lane_id), owner=slab)
        src = cuda.from_cuda_array_interface(_cuda_array_interface_for_array(value), owner=value)
        dst.copy_to_device(src)
        slab.array.block_until_ready()
        return slab


class LocalDeviceSlabBackend(DeviceSlabBackend):
    transport = "local-device"

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        array = jnp.zeros(spec.slab_shape, dtype=np.dtype(spec.dtype))
        array.block_until_ready()
        handle = DeviceSlabHandle(
            spec=spec,
            transport=self.transport,
            device_ordinal=_array_device_ordinal(array),
            handle_bytes=b"",
            strides=_c_contiguous_strides(spec.slab_shape, np.dtype(spec.dtype)),
        )
        return DeviceSlab(spec, array, handle)

    def open_slab(self, handle: DeviceSlabHandle) -> DeviceSlab:
        raise RuntimeError("LocalDeviceSlabBackend cannot open slabs across processes")


def make_default_device_slab_backend() -> DeviceSlabBackend:
    if _has_cuda_device():
        return CudaIpcDeviceSlabBackend()
    return LocalDeviceSlabBackend()


def has_cuda_device() -> bool:
    return _has_cuda_device()


def _export_cuda_ipc_handle(spec: DeviceSlabSpec, array: jax.Array) -> DeviceSlabHandle:
    device = next(iter(array.devices()))
    cuda.select_device(device.id)
    context = cuda.current_context()
    device_pointer = drvapi.cu_device_ptr(array.unsafe_buffer_pointer())
    memory = driver.MemoryPointer(
        context,
        device_pointer,
        int(array.size * array.dtype.itemsize),
        owner=array,
    )
    ipc_handle = context.get_ipc_handle(memory)
    if driver.USE_NV_BINDING:
        handle_bytes = bytes(ipc_handle.handle.reserved)
    else:
        handle_bytes = bytes(ipc_handle.handle)
    return DeviceSlabHandle(
        spec=spec,
        transport=CudaIpcDeviceSlabBackend.transport,
        device_ordinal=device.id,
        handle_bytes=handle_bytes,
        ready_event_bytes=None,
        offset=int(ipc_handle.offset),
        strides=_c_contiguous_strides(spec.slab_shape, np.dtype(spec.dtype)),
    )


def _validate_lane_update(slab: DeviceSlab, lane_id: int, value: jax.Array) -> None:
    if lane_id < 0 or lane_id >= slab.spec.max_lanes:
        raise ValueError(f"lane_id {lane_id} outside slab capacity {slab.spec.max_lanes}")
    expected_shape = (1, *slab.spec.shape[1:])
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"lane update shape must be {expected_shape}, got {tuple(value.shape)}")
    if np.dtype(value.dtype) != np.dtype(slab.spec.dtype):
        raise ValueError(f"lane update dtype must be {slab.spec.dtype}, got {value.dtype}")


def _is_gpu_array(array: jax.Array) -> bool:
    return len(array.devices()) == 1 and next(iter(array.devices())).platform == "gpu"


def _array_device_ordinal(array: jax.Array) -> int:
    device = next(iter(array.devices()))
    return int(getattr(device, "id", 0))


def _has_cuda_device() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def _c_contiguous_strides(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
    stride = dtype.itemsize
    strides: list[int] = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _cuda_array_interface_for_array(array: jax.Array) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    return {
        "shape": tuple(array.shape),
        "strides": _c_contiguous_strides(tuple(array.shape), dtype),
        "typestr": dtype.str,
        "data": (array.unsafe_buffer_pointer(), False),
        "version": 3,
    }


def _cuda_array_interface_for_lane(array: jax.Array, spec: DeviceSlabSpec, lane_id: int) -> dict[str, Any]:
    dtype = np.dtype(spec.dtype)
    lane_shape = (1, *spec.shape[1:])
    lane_nbytes = int(np.prod(lane_shape) * dtype.itemsize)
    return {
        "shape": lane_shape,
        "strides": _c_contiguous_strides(lane_shape, dtype),
        "typestr": dtype.str,
        "data": (array.unsafe_buffer_pointer() + lane_id * lane_nbytes, False),
        "version": 3,
    }


@contextlib.contextmanager
def opened_slab(handle: DeviceSlabHandle, backend: DeviceSlabBackend | None = None) -> Iterator[DeviceSlab]:
    slab = (backend or make_default_device_slab_backend()).open_slab(handle)
    try:
        yield slab
    finally:
        slab.close()
