from __future__ import annotations

import dataclasses
import contextlib
import json
import multiprocessing as mp
import os
import pathlib
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv import drvapi

from openpi.serving.va_split_jax.device_slab import _numba_device_ordinal_for_jax_device
from openpi.serving.va_split_jax.device_slab import _select_numba_device


LOG_PATH = pathlib.Path("logs/tests/jax_device_ipc_gate.json")


@dataclasses.dataclass(frozen=True, slots=True)
class GateReport:
    status: str
    reason: str
    producer_devices: list[str]
    consumer_devices: list[str]
    payload_shape: tuple[int, ...]
    payload_dtype: str
    elapsed_ms: float
    transport: str


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceIpcHandle:
    transport: str
    device_ordinal: int
    shape: tuple[int, ...]
    dtype: str
    handle_bytes: bytes
    event_handle_bytes: bytes | None
    offset: int = 0
    strides: tuple[int, ...] | None = None


def _write_report(report: GateReport) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True), encoding="utf-8")


def _device_summary() -> list[str]:
    return [str(device) for device in jax.devices()]


def _has_cuda_device() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def _c_contiguous_strides(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
    stride = dtype.itemsize
    strides: list[int] = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _producer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    start_ns = time.monotonic_ns()
    try:
        payload = jnp.arange(1024, dtype=jnp.float32).reshape(16, 64)
        payload.block_until_ready()
        # 第一版门禁只接受真正的 device-side IPC handle。
        # 具体 handle 导出函数由后续 Step 3 填入，不允许退化成 np.asarray(payload)。
        handle = _export_device_ipc_handle(payload)
        control_queue.put(
            {
                "handle": handle,
                "shape": tuple(payload.shape),
                "dtype": str(payload.dtype),
                "producer_devices": _device_summary(),
                "start_ns": start_ns,
            }
        )
        ack = result_queue.get(timeout=30)
        if ack != "ok":
            raise RuntimeError(f"consumer returned {ack!r}")
    except Exception as exc:
        control_queue.put({"error": repr(exc), "producer_devices": _device_summary(), "start_ns": start_ns})
        raise


def _consumer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    message = control_queue.get(timeout=30)
    if "error" in message:
        result_queue.put(f"producer-error: {message['error']}")
        return
    try:
        with _open_device_ipc_handle(message["handle"], message["shape"], message["dtype"]) as array:
            actual = np.asarray(array)
            expected = np.arange(1024, dtype=np.float32).reshape(16, 64)
            np.testing.assert_allclose(actual, expected)
        result_queue.put("ok")
    except Exception as exc:
        result_queue.put(f"consumer-error: {exc!r}")


def _export_device_ipc_handle(array: jax.Array) -> Any:
    if len(array.devices()) != 1:
        raise RuntimeError(f"expected a single-device JAX array, got devices={array.devices()!r}")
    device = next(iter(array.devices()))
    if device.platform != "gpu":
        raise RuntimeError(f"expected a GPU-backed JAX array, got {device!r}")

    array.block_until_ready()
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
    if driver.USE_NV_BINDING:
        handle_bytes = bytes(ipc_handle.handle.reserved)
    else:
        handle_bytes = bytes(ipc_handle.handle)
    return DeviceIpcHandle(
        transport="cuda-ipc-numba",
        device_ordinal=device_ordinal,
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        handle_bytes=handle_bytes,
        event_handle_bytes=None,
        offset=int(ipc_handle.offset),
        strides=_c_contiguous_strides(tuple(array.shape), np.dtype(array.dtype)),
    )


def _import_device_ipc_handle(handle: Any, shape: tuple[int, ...], dtype: str) -> jax.Array:
    with _open_device_ipc_handle(handle, shape, dtype) as array:
        return array


@contextlib.contextmanager
def _open_device_ipc_handle(handle: Any, shape: tuple[int, ...], dtype: str):
    if not isinstance(handle, DeviceIpcHandle):
        raise TypeError(f"expected DeviceIpcHandle, got {type(handle)!r}")
    if handle.transport != "cuda-ipc-numba":
        raise RuntimeError(f"unsupported IPC transport {handle.transport!r}")
    if tuple(shape) != handle.shape:
        raise RuntimeError(f"shape mismatch: message={shape!r}, handle={handle.shape!r}")
    if str(np.dtype(dtype)) != str(np.dtype(handle.dtype)):
        raise RuntimeError(f"dtype mismatch: message={dtype!r}, handle={handle.dtype!r}")

    _select_numba_device(handle.device_ordinal)
    with cuda.open_ipc_array(
        tuple(handle.handle_bytes),
        handle.shape,
        np.dtype(handle.dtype),
        strides=handle.strides,
        offset=handle.offset,
    ) as ipc_array:
        array = jnp.asarray(ipc_array)
        array.block_until_ready()
        if array.unsafe_buffer_pointer() != ipc_array.device_ctypes_pointer.value:
            raise RuntimeError("JAX imported IPC memory by copy instead of aliasing the opened device allocation")
        yield array


def test_jax_device_ipc_gate_requires_cross_process_device_handle():
    start = time.monotonic()
    devices = _device_summary()
    if not _has_cuda_device():
        report = GateReport(
            status="skipped",
            reason="No CUDA JAX device is visible; device IPC gate requires GPU.",
            producer_devices=devices,
            consumer_devices=[],
            payload_shape=(16, 64),
            payload_dtype="float32",
            elapsed_ms=0.0,
            transport="none",
        )
        _write_report(report)
        pytest.skip(report.reason)

    ctx = mp.get_context("spawn")
    control_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    producer = ctx.Process(target=_producer, args=(control_queue, result_queue))
    consumer = ctx.Process(target=_consumer, args=(control_queue, result_queue))
    producer.start()
    consumer.start()
    producer.join(timeout=45)
    consumer.join(timeout=45)

    elapsed_ms = (time.monotonic() - start) * 1000
    status = "passed" if producer.exitcode == 0 and consumer.exitcode == 0 else "failed"
    report = GateReport(
        status=status,
        reason=f"producer_exit={producer.exitcode}, consumer_exit={consumer.exitcode}",
        producer_devices=devices,
        consumer_devices=devices,
        payload_shape=(16, 64),
        payload_dtype="float32",
        elapsed_ms=elapsed_ms,
        transport="device-ipc",
    )
    _write_report(report)

    assert producer.exitcode == 0
    assert consumer.exitcode == 0
