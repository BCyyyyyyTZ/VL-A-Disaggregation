from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import os
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from numba import cuda
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.cuda.cudadrv import devicearray
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv import drvapi
import numpy as np


@dataclass(frozen=True, slots=True)
class DeviceSlabSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    max_lanes: int
    lane_axis: int = 0

    @property
    def slab_shape(self) -> tuple[int, ...]:
        shape = list(self.shape)
        shape[self.normalized_lane_axis] = self.max_lanes
        return tuple(shape)

    @property
    def normalized_lane_axis(self) -> int:
        lane_axis = self.lane_axis
        if lane_axis < 0:
            lane_axis += len(self.shape)
        if lane_axis < 0 or lane_axis >= len(self.shape):
            raise ValueError(f"lane_axis {self.lane_axis} outside shape rank {len(self.shape)}")
        return lane_axis


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

    def copy_lane_from_array(
        self,
        slab: DeviceSlab,
        lane_id: int,
        value: jax.Array,
        *,
        sync: bool = True,
    ) -> DeviceSlab:
        del sync  # JAX local path has no write stream; always complete the update.
        _validate_lane_update(slab, lane_id, value)
        value = _storage_value_for_spec(value, slab.spec)
        update = jax.lax.dynamic_update_slice_in_dim(
            slab.array, value, lane_id, axis=slab.spec.normalized_lane_axis
        )
        update.block_until_ready()
        return DeviceSlab(slab.spec, update, slab.handle, slab._close_stack)  # noqa: SLF001

    def copy_batch_from_array(
        self,
        slab: DeviceSlab,
        lane_start: int,
        value: jax.Array,
        *,
        sync: bool = True,
    ) -> DeviceSlab:
        del sync  # JAX local path has no write stream; always complete the update.
        _validate_batch_update(slab, lane_start, value)
        value = _storage_value_for_spec(value, slab.spec)
        update = jax.lax.dynamic_update_slice_in_dim(
            slab.array, value, lane_start, axis=slab.spec.normalized_lane_axis
        )
        update.block_until_ready()
        return DeviceSlab(slab.spec, update, slab.handle, slab._close_stack)  # noqa: SLF001

    def sync_write_stream(self, device_ordinal: int | None = None) -> None:
        """No-op for backends that complete each lane copy synchronously."""
        del device_ordinal

    def view_batch(self, slab: DeviceSlab, batch_size: int) -> jax.Array:
        if batch_size < 0 or batch_size > slab.spec.max_lanes:
            raise ValueError(f"batch_size {batch_size} outside slab capacity {slab.spec.max_lanes}")
        return _logical_view_for_spec(
            jax.lax.dynamic_slice_in_dim(slab.array, 0, batch_size, axis=slab.spec.normalized_lane_axis), slab.spec
        )

    def slice_lanes(self, slab: DeviceSlab, slot_ids: tuple[int, ...]) -> jax.Array:
        """Return a dense-prefix device-side view for already-mapped lanes without reopening IPC."""
        if slot_ids != tuple(range(len(slot_ids))):
            raise ValueError("The first implementation only permits dense-prefix slot ids")
        return self.view_batch(slab, len(slot_ids))


class CudaIpcDeviceSlabBackend(DeviceSlabBackend):
    transport = "cuda-ipc-numba"

    def __init__(self, *, device_ordinal: int | None = None):
        self._device_ordinal = device_ordinal
        self._write_streams: dict[int, Any] = {}

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        if spec.max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        dtype = _storage_dtype_for_spec(spec)
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
        device_ordinal = _normalize_numba_device_ordinal(handle.device_ordinal)
        context = _select_numba_device(device_ordinal)
        stack = contextlib.ExitStack()
        try:
            driver_handle = (
                driver.binding.CUipcMemHandle() if driver.USE_NV_BINDING else driver.drvapi.cu_ipc_mem_handle()
            )
            if driver.USE_NV_BINDING:
                driver_handle.reserved = tuple(handle.handle_bytes)
            else:
                driver_handle = driver.drvapi.cu_ipc_mem_handle(*tuple(handle.handle_bytes))
            ipchandle = driver.IpcHandle(
                None,
                driver_handle,
                int(np.prod(handle.spec.slab_shape) * np.dtype(_storage_dtype_for_spec(handle.spec)).itemsize),
                offset=handle.offset,
            )
            ipc_array = ipchandle.open_array(
                context,
                shape=handle.spec.slab_shape,
                dtype=_storage_dtype_for_spec(handle.spec),
                strides=handle.strides,
            )
            stack.callback(ipchandle.close)
            array = jnp.asarray(ipc_array)
            array.block_until_ready()
            if array.unsafe_buffer_pointer() != ipc_array.device_ctypes_pointer.value:
                raise RuntimeError("JAX imported IPC slab by copy instead of aliasing the opened device allocation")
            return DeviceSlab(handle.spec, array, handle, stack)
        except Exception:
            stack.close()
            raise

    def copy_lane_from_array(
        self,
        slab: DeviceSlab,
        lane_id: int,
        value: jax.Array,
        *,
        sync: bool = True,
    ) -> DeviceSlab:
        """Copy one lane into an IPC slab without device-wide or whole-slab sync.

        Only waits for ``value`` (producer output) and, when ``sync=True``, the
        dedicated write stream that ran this copy. Free-lane writes can proceed
        while another process reads other lanes of the same slab.
        """
        _validate_lane_update(slab, lane_id, value)
        if slab.handle.transport != self.transport:
            raise RuntimeError(f"expected {self.transport!r} slab, got {slab.handle.transport!r}")
        value.block_until_ready()
        device_ordinal = _normalize_numba_device_ordinal(slab.handle.device_ordinal)
        _select_numba_device(device_ordinal)
        stream = self._write_stream(device_ordinal)
        dst = _device_array_view_from_cuda_array_interface(
            _cuda_array_interface_for_array(slab.array), owner=slab, device_ordinal=device_ordinal
        )
        src = _device_array_view_from_cuda_array_interface(
            _cuda_array_interface_for_array(value), owner=value, device_ordinal=device_ordinal
        )
        _copy_lane_with_driver(dst, src, lane_id, slab.spec, stream=stream)
        if sync:
            stream.synchronize()
        return slab

    def copy_batch_from_array(
        self,
        slab: DeviceSlab,
        lane_start: int,
        value: jax.Array,
        *,
        sync: bool = True,
    ) -> DeviceSlab:
        """Copy a contiguous batch into an IPC slab with one kernel per leaf."""
        _validate_batch_update(slab, lane_start, value)
        if slab.handle.transport != self.transport:
            raise RuntimeError(f"expected {self.transport!r} slab, got {slab.handle.transport!r}")
        value.block_until_ready()
        device_ordinal = _normalize_numba_device_ordinal(slab.handle.device_ordinal)
        _select_numba_device(device_ordinal)
        stream = self._write_stream(device_ordinal)
        dst = _device_array_view_from_cuda_array_interface(
            _cuda_array_interface_for_array(slab.array), owner=slab, device_ordinal=device_ordinal
        )
        src = _device_array_view_from_cuda_array_interface(
            _cuda_array_interface_for_array(value), owner=value, device_ordinal=device_ordinal
        )
        _copy_batch_with_driver(dst, src, lane_start, slab.spec, stream=stream)
        if sync:
            stream.synchronize()
        return slab

    def sync_write_stream(self, device_ordinal: int | None = None) -> None:
        """Synchronize the dedicated lane-write stream (not the whole device)."""
        ordinal = self._device_ordinal if device_ordinal is None else device_ordinal
        if ordinal is None:
            for stream in self._write_streams.values():
                stream.synchronize()
            return
        stream = self._write_streams.get(int(ordinal))
        if stream is not None:
            stream.synchronize()

    def _write_stream(self, device_ordinal: int):
        key = _normalize_numba_device_ordinal(device_ordinal)
        stream = self._write_streams.get(key)
        if stream is None:
            context = _select_numba_device(key)
            stream = context.create_stream()
            self._write_streams[key] = stream
        return stream


class LocalDeviceSlabBackend(DeviceSlabBackend):
    transport = "local-device"

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        array = jnp.zeros(spec.slab_shape, dtype=_storage_dtype_for_spec(spec))
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


def make_default_device_slab_backend(*, device_ordinal: int | None = None) -> DeviceSlabBackend:
    if _has_cuda_device():
        return CudaIpcDeviceSlabBackend(device_ordinal=device_ordinal)
    return LocalDeviceSlabBackend()


def has_cuda_device() -> bool:
    return _has_cuda_device()


def _export_cuda_ipc_handle(spec: DeviceSlabSpec, array: jax.Array) -> DeviceSlabHandle:
    device = next(iter(array.devices()))
    device_ordinal = _numba_device_ordinal_for_jax_device(device)
    context = _select_numba_device(device_ordinal)
    device_pointer = drvapi.cu_device_ptr(array.unsafe_buffer_pointer())
    memory = driver.MemoryPointer(
        context,
        device_pointer,
        int(array.size * array.dtype.itemsize),
        owner=array,
    )
    ipc_handle = context.get_ipc_handle(memory)
    handle_bytes = bytes(ipc_handle.handle.reserved) if driver.USE_NV_BINDING else bytes(ipc_handle.handle)
    return DeviceSlabHandle(
        spec=spec,
        transport=CudaIpcDeviceSlabBackend.transport,
        device_ordinal=device_ordinal,
        handle_bytes=handle_bytes,
        ready_event_bytes=None,
        offset=int(ipc_handle.offset),
        strides=_c_contiguous_strides(spec.slab_shape, _storage_dtype_for_spec(spec)),
    )


def _validate_lane_update(slab: DeviceSlab, lane_id: int, value: jax.Array) -> None:
    if lane_id < 0 or lane_id >= slab.spec.max_lanes:
        raise ValueError(f"lane_id {lane_id} outside slab capacity {slab.spec.max_lanes}")
    expected_shape = slab.spec.shape
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"lane update shape must be {expected_shape}, got {tuple(value.shape)}")
    if np.dtype(value.dtype) != np.dtype(slab.spec.dtype):
        raise ValueError(f"lane update dtype must be {slab.spec.dtype}, got {value.dtype}")


def _validate_batch_update(slab: DeviceSlab, lane_start: int, value: jax.Array) -> None:
    axis = slab.spec.normalized_lane_axis
    if lane_start < 0 or lane_start >= slab.spec.max_lanes:
        raise ValueError(f"lane_start {lane_start} outside slab capacity {slab.spec.max_lanes}")
    batch_lanes = int(value.shape[axis])
    if batch_lanes <= 0:
        raise ValueError("batch update must contain at least one lane")
    if lane_start + batch_lanes > slab.spec.max_lanes:
        raise ValueError(
            f"batch update lanes [{lane_start}, {lane_start + batch_lanes}) exceed slab capacity "
            f"{slab.spec.max_lanes}"
        )
    expected_shape = list(slab.spec.shape)
    expected_shape[axis] = batch_lanes
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"batch update shape must be {tuple(expected_shape)}, got {tuple(value.shape)}")
    if np.dtype(value.dtype) != np.dtype(slab.spec.dtype):
        raise ValueError(f"batch update dtype must be {slab.spec.dtype}, got {value.dtype}")


def _is_gpu_array(array: jax.Array) -> bool:
    return len(array.devices()) == 1 and next(iter(array.devices())).platform == "gpu"


def _array_device_ordinal(array: jax.Array) -> int:
    device = next(iter(array.devices()))
    return _numba_device_ordinal_for_jax_device(device)


def _select_numba_device(device_ordinal: int):
    """Select a Numba CUDA context using process-visible device ordinal."""
    return cuda.current_context(_normalize_numba_device_ordinal(device_ordinal))


def _device_array_view_from_cuda_array_interface(
    desc: dict[str, Any], *, owner: Any, device_ordinal: int
) -> devicearray.DeviceNDArray:
    """Create a Numba device array view in the selected process-visible context."""
    version = int(desc.get("version", 0))
    if version >= 1 and desc.get("mask") is not None:
        raise NotImplementedError("Masked arrays are not supported")

    shape = desc["shape"]
    strides = desc.get("strides")
    dtype = np.dtype(desc["typestr"])
    shape, strides, dtype = prepare_shape_strides_dtype(shape, strides, dtype, order="C")
    size = driver.memory_size_from_info(shape, strides, dtype.itemsize)
    context = _select_numba_device(device_ordinal)
    devptr = driver.get_devptr_for_active_ctx(desc["data"][0])
    data = driver.MemoryPointer(context, devptr, size=size, owner=owner)
    stream_ptr = desc.get("stream")
    stream = context.create_external_stream(stream_ptr) if stream_ptr is not None else 0
    return devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=data, stream=stream)


def _normalize_numba_device_ordinal(device_ordinal: int) -> int:
    ordinal = int(device_ordinal)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if visible_tokens:
        if len(visible_tokens) == 1:
            return 0
        for visible_ordinal, token in enumerate(visible_tokens):
            if token.isdigit() and int(token) == ordinal:
                return visible_ordinal

    try:
        numba_device_count = len(cuda.gpus)
    except Exception:
        numba_device_count = 0
    if 0 <= ordinal < numba_device_count:
        return ordinal
    if numba_device_count == 1:
        return 0
    raise RuntimeError(
        "Cannot map CUDA device ordinal to a Numba process-visible ordinal: "
        f"device_ordinal={device_ordinal} numba_device_count={numba_device_count} "
        f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
    )


def _numba_device_ordinal_for_jax_device(device: jax.Device) -> int:
    """Map a JAX device to the ordinal understood by Numba in this process.

    JAX/XLA device ids can reflect physical GPU ids in some CUDA_VISIBLE_DEVICES
    setups, while Numba cuda.select_device() indexes the process-visible device
    list. With CUDA_VISIBLE_DEVICES=5 or CUDA_VISIBLE_DEVICES=GPU-..., a single
    visible GPU must be selected as ordinal 0 even if JAX reports a different id.
    """
    jax_device_id = int(getattr(device, "id", 0))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if visible_tokens:
        if len(visible_tokens) == 1:
            return 0
        for ordinal, token in enumerate(visible_tokens):
            if token.isdigit() and int(token) == jax_device_id:
                return ordinal

    try:
        numba_device_count = len(cuda.gpus)
    except Exception:
        numba_device_count = 0

    if 0 <= jax_device_id < numba_device_count:
        return jax_device_id

    if numba_device_count == 1:
        return 0

    raise RuntimeError(
        "Cannot map JAX GPU device to a Numba CUDA ordinal: "
        f"jax_device={device!r} jax_device_id={jax_device_id} "
        f"numba_device_count={numba_device_count} CUDA_VISIBLE_DEVICES={visible_devices!r}"
    )


def _has_cuda_device() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def _c_contiguous_strides(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
    stride = dtype.itemsize
    strides: list[int] = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _storage_dtype_for_spec(spec: DeviceSlabSpec) -> np.dtype:
    return _storage_dtype_for_jax_dtype(spec.dtype)


def _storage_dtype_for_jax_dtype(dtype: Any) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(jnp.bfloat16):
        return np.dtype(np.uint16)
    return dtype


def _storage_value_for_spec(value: jax.Array, spec: DeviceSlabSpec) -> jax.Array:
    if np.dtype(spec.dtype) == np.dtype(jnp.bfloat16):
        return jax.lax.bitcast_convert_type(value, jnp.uint16)
    return value


def _logical_view_for_spec(array: jax.Array, spec: DeviceSlabSpec) -> jax.Array:
    if np.dtype(spec.dtype) == np.dtype(jnp.bfloat16):
        return jax.lax.bitcast_convert_type(array, jnp.bfloat16)
    return array


def _cuda_array_interface_for_array(array: jax.Array) -> dict[str, Any]:
    dtype = _storage_dtype_for_jax_dtype(array.dtype)
    return {
        "shape": tuple(array.shape),
        "strides": _c_contiguous_strides(tuple(array.shape), dtype),
        "typestr": dtype.str,
        "data": (array.unsafe_buffer_pointer(), False),
        "version": 3,
    }


def _cuda_array_interface_for_lane(array: jax.Array, spec: DeviceSlabSpec, lane_id: int) -> dict[str, Any]:
    dtype = _storage_dtype_for_spec(spec)
    lane_shape = spec.shape
    slab_strides = _c_contiguous_strides(spec.slab_shape, dtype)
    lane_offset_bytes = lane_id * slab_strides[spec.normalized_lane_axis]
    return {
        "shape": lane_shape,
        "strides": slab_strides,
        "typestr": dtype.str,
        "data": (array.unsafe_buffer_pointer() + lane_offset_bytes, False),
        "version": 3,
    }


def _copy_lane_with_driver(dst: Any, src: Any, lane_id: int, spec: DeviceSlabSpec, *, stream: Any) -> None:
    axis = spec.normalized_lane_axis
    inner_elems = _prod(spec.shape[axis + 1 :])
    outer_elems = _prod(spec.shape[:axis])
    itemsize = _storage_dtype_for_spec(spec).itemsize
    segment_bytes = inner_elems * itemsize
    for outer in range(outer_elems):
        dst_offset = (outer * spec.max_lanes * inner_elems + lane_id * inner_elems) * itemsize
        src_offset = outer * inner_elems * itemsize
        driver.device_to_device(
            dst.gpu_data.view(dst_offset, dst_offset + segment_bytes),
            src.gpu_data.view(src_offset, src_offset + segment_bytes),
            segment_bytes,
            stream=stream,
        )


def _copy_batch_with_driver(dst: Any, src: Any, lane_start: int, spec: DeviceSlabSpec, *, stream: Any) -> None:
    axis = spec.normalized_lane_axis
    batch_lanes = int(src.shape[axis])
    inner_elems = _prod(spec.shape[axis + 1 :])
    outer_elems = _prod(spec.shape[:axis])
    itemsize = _storage_dtype_for_spec(spec).itemsize
    segment_bytes = inner_elems * itemsize
    for outer in range(outer_elems):
        for lane in range(batch_lanes):
            dst_offset = (outer * spec.max_lanes * inner_elems + (lane_start + lane) * inner_elems) * itemsize
            src_offset = (outer * batch_lanes * inner_elems + lane * inner_elems) * itemsize
            driver.device_to_device(
                dst.gpu_data.view(dst_offset, dst_offset + segment_bytes),
                src.gpu_data.view(src_offset, src_offset + segment_bytes),
                segment_bytes,
                stream=stream,
            )


def _prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _copy_lane_with_kernel(
    dst: Any,
    src: Any,
    lane_id: int,
    spec: DeviceSlabSpec,
    *,
    stream: Any | None = None,
) -> None:
    axis = spec.normalized_lane_axis
    inner_elems = int(np.prod(spec.shape[axis + 1 :], dtype=np.int64))
    total_elems = int(np.prod(spec.shape, dtype=np.int64))
    threads_per_block = 256
    blocks = (total_elems + threads_per_block - 1) // threads_per_block
    if stream is None:
        _copy_lane_kernel[blocks, threads_per_block](
            dst, src, total_elems, lane_id, spec.max_lanes, inner_elems
        )
    else:
        _copy_lane_kernel[blocks, threads_per_block, stream](
            dst, src, total_elems, lane_id, spec.max_lanes, inner_elems
        )


def _copy_batch_with_kernel(
    dst: Any,
    src: Any,
    lane_start: int,
    spec: DeviceSlabSpec,
    *,
    stream: Any | None = None,
) -> None:
    axis = spec.normalized_lane_axis
    batch_lanes = int(src.shape[axis])
    inner_elems = int(np.prod(spec.shape[axis + 1 :], dtype=np.int64))
    total_elems = int(np.prod(src.shape, dtype=np.int64))
    threads_per_block = 256
    blocks = (total_elems + threads_per_block - 1) // threads_per_block
    if stream is None:
        _copy_batch_kernel[blocks, threads_per_block](
            dst, src, total_elems, lane_start, spec.max_lanes, batch_lanes, inner_elems
        )
    else:
        _copy_batch_kernel[blocks, threads_per_block, stream](
            dst, src, total_elems, lane_start, spec.max_lanes, batch_lanes, inner_elems
        )


@cuda.jit
def _copy_lane_kernel(dst, src, total_elems, lane_id, lane_axis_size, inner_elems):  # pragma: no cover
    index = cuda.grid(1)
    if index >= total_elems:
        return
    outer = index // inner_elems
    inner = index - outer * inner_elems
    dst_index = outer * lane_axis_size * inner_elems + lane_id * inner_elems + inner
    dst.flat[dst_index] = src.flat[index]


@cuda.jit
def _copy_batch_kernel(  # pragma: no cover
    dst, src, total_elems, lane_start, lane_axis_size, batch_lanes, inner_elems
):
    index = cuda.grid(1)
    if index >= total_elems:
        return
    outer = index // (batch_lanes * inner_elems)
    rem = index - outer * batch_lanes * inner_elems
    lane = rem // inner_elems
    inner = rem - lane * inner_elems
    dst_index = outer * lane_axis_size * inner_elems + (lane_start + lane) * inner_elems + inner
    dst.flat[dst_index] = src.flat[index]


@contextlib.contextmanager
def opened_slab(handle: DeviceSlabHandle, backend: DeviceSlabBackend | None = None) -> Iterator[DeviceSlab]:
    slab = (backend or make_default_device_slab_backend()).open_slab(handle)
    try:
        yield slab
    finally:
        slab.close()
