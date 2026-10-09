"""TCP control plane for multi-machine VLM–AE split.

Host shared memory cannot cross machines. This transport keeps the same credit,
FCFS, overlap-export, and packed-noise schedule, and moves prefix rows as
numpy payloads.
"""

from __future__ import annotations

from collections import deque
from multiprocessing.connection import Client
from multiprocessing.connection import Listener
import os
import queue
import threading
import time
from typing import Any

from openpi.serving.va_split_multinode.messages import PrefixMsg
from openpi.serving.va_split_multinode.messages import RunConfig

AUTHKEY = b"va-split-multinode"


class CreditHub:
    def __init__(self, count: int) -> None:
        self._ids: deque[int] = deque(range(int(count)))
        self._cv = threading.Condition()
        self._closed = False

    def acquire(self, count: int, *, timeout_s: float) -> list[int]:
        deadline = time.perf_counter() + timeout_s
        with self._cv:
            while not self._ids and not self._closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a prefix-lane credit")
                self._cv.wait(timeout=remaining)
            if self._closed and not self._ids:
                raise RuntimeError("credit hub is closed")
            granted: list[int] = []
            while self._ids and len(granted) < count:
                granted.append(self._ids.popleft())
            return granted

    def release(self, lane_ids: list[int]) -> None:
        with self._cv:
            self._ids.extend(int(lane_id) for lane_id in lane_ids)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class ReleaseQueue:
    """AE-side credit queue: `put` returns lanes to the hub."""

    def __init__(self, hub: CreditHub) -> None:
        self._hub = hub

    def put(self, lane_ids: list[int]) -> None:
        self._hub.release(list(lane_ids))


class Duplex:
    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._out: queue.Queue[Any] = queue.Queue()
        self._in: queue.Queue[Any] = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, name="va-tcp-write", daemon=True)
        self._reader = threading.Thread(target=self._read_loop, name="va-tcp-read", daemon=True)
        self._writer.start()
        self._reader.start()

    def send(self, message: Any) -> None:
        self._out.put(message)

    def recv(self, timeout: float | None = None) -> Any:
        message = self._in.get(timeout=timeout)
        if isinstance(message, BaseException):
            raise message
        return message

    def close(self) -> None:
        self._out.put(None)
        self._writer.join(timeout=5)

    def _write_loop(self) -> None:
        try:
            while True:
                message = self._out.get()
                if message is None:
                    with _ignore():
                        self._conn.send({"op": "bye"})
                    return
                self._conn.send(message)
        except Exception as exc:
            self._in.put(exc)

    def _read_loop(self) -> None:
        try:
            while True:
                message = self._conn.recv()
                if isinstance(message, dict) and message.get("op") == "bye":
                    self._in.put(EOFError("peer closed"))
                    return
                self._in.put(message)
        except EOFError as exc:
            self._in.put(exc)
        except Exception as exc:
            self._in.put(exc)


class RpcCredit:
    def __init__(self, duplex: Duplex, *, max_batch_size: int, timeout_s: float) -> None:
        self._duplex = duplex
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_s

    def get(self) -> list[int]:
        self._duplex.send({"op": "pull", "n": self._max_batch_size})
        message = self._duplex.recv(timeout=self._timeout_s)
        if not isinstance(message, dict) or message.get("op") != "credit":
            raise RuntimeError(f"expected credit reply, got {message!r}")
        return [int(lane_id) for lane_id in message["ids"]]

    def put(self, lane_ids: list[int]) -> None:
        if lane_ids:
            self._duplex.send({"op": "release", "ids": [int(lane_id) for lane_id in lane_ids]})


class PrefixSender:
    def __init__(self, duplex: Duplex) -> None:
        self._duplex = duplex

    def put(self, message: PrefixMsg | None) -> None:
        if message is None:
            self._duplex.send({"op": "close"})
            return
        self._duplex.send({"op": "prefix", "msg": _prefix_to_wire(message)})


def connect(addr: str, *, timeout_s: float) -> Any:
    host, port = _split_addr(addr)
    deadline = time.perf_counter() + timeout_s
    last: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            return Client((host, port), authkey=AUTHKEY)
        except (OSError, EOFError, ConnectionError) as exc:
            last = exc
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to {addr}") from last


def split_bind(bind_host: str, addr: str = "") -> tuple[str, int]:
    """Accept ``host`` or ``host:port``. An explicit address supplies the port when bind has none."""
    if bind_host.count(":") == 1 and bind_host.rsplit(":", 1)[1].isdigit():
        host, port = bind_host.rsplit(":", 1)
        return host, int(port)
    return bind_host, _port(addr)


def listen(host: str, port: int = 0) -> tuple[Listener, str]:
    listener = Listener((host, int(port)), authkey=AUTHKEY)
    bound_port = int(listener.address[1])
    connect_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    return listener, f"{connect_host}:{bound_port}"


def prepare_worker_env(cfg: RunConfig, device: int) -> None:
    if cfg.launch != "process":
        return
    if cfg.backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if device >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["JAX_PLATFORMS"] = "cpu"
    elif cfg.backend == "pytorch":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def serve_ae(cfg: RunConfig, *, result_addr: str, bind_host: str) -> None:
    from openpi.serving.va_split_multinode.workers import run_ae_worker

    prepare_worker_env(cfg, cfg.ae_device)
    result_conn = connect(result_addr, timeout_s=cfg.startup_timeout_s)
    result = Duplex(result_conn)
    listener, prefix_addr = listen(*split_bind(bind_host, cfg.ae_addr))
    result.send({"kind": "listen", "role": "ae", "addr": prefix_addr})
    hub = CreditHub(cfg.max_prefix_slots)
    prefix_q: queue.Queue[Any] = queue.Queue()
    example: dict[str, Any] = {}
    example_ready = threading.Event()
    stop = threading.Event()

    def _accept_loop() -> None:
        accepted = 0
        while accepted < len(cfg.vlm_devices):
            conn = listener.accept()
            accepted += 1
            duplex = Duplex(conn)
            threading.Thread(target=_ae_peer, args=(duplex, hub, prefix_q, example, example_ready, cfg), daemon=True).start()
        listener.close()

    threading.Thread(target=_accept_loop, name="va-ae-accept", daemon=True).start()

    def _wait_example() -> dict[str, Any]:
        if not example_ready.wait(timeout=cfg.startup_timeout_s):
            raise TimeoutError("AE did not receive a prefix example row")
        return example["row"]

    result_q: queue.Queue[Any] = queue.Queue()

    def _pump() -> None:
        while not stop.is_set():
            try:
                message = result_q.get(timeout=0.2)
            except queue.Empty:
                continue
            result.send(message)
            if message.get("kind") in {"done", "worker_error"}:
                return

    pump = threading.Thread(target=_pump, name="va-ae-results", daemon=True)
    pump.start()
    try:
        run_ae_worker(
            cfg,
            device=cfg.ae_device,
            num_vlm=len(cfg.vlm_devices),
            prefix_q=prefix_q,
            result_q=result_q,
            pool_q=None,
            credit_q=ReleaseQueue(hub),
            example_wait=_wait_example,
        )
    finally:
        stop.set()
        hub.close()
        pump.join(timeout=cfg.timeout_s)
        result.close()


def serve_vlm(cfg: RunConfig, *, bind_host: str) -> None:
    from openpi.serving.va_split_multinode.workers import run_vlm_worker

    device = cfg.vlm_devices[cfg.worker_index]
    prepare_worker_env(cfg, device)
    ae_conn = connect(cfg.ae_addr, timeout_s=cfg.startup_timeout_s)
    duplex = Duplex(ae_conn)
    vlm_addr = cfg.vlm_addrs[cfg.worker_index] if cfg.vlm_addrs else ""
    listener, work_addr = listen(*split_bind(bind_host, vlm_addr))
    work_q: queue.Queue[Any] = queue.Queue()
    if cfg.extra.get("address_queue") is not None:
        cfg.extra["address_queue"].put(work_addr)

    def _on_ready(backend: Any) -> None:
        if cfg.worker_index == 0:
            duplex.send({"op": "example", "row": backend.example_row()})

    def _accept_work() -> None:
        conn = listener.accept()
        peer = Duplex(conn)
        try:
            while True:
                try:
                    message = peer.recv(timeout=1.0)
                except queue.Empty:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("op") == "work":
                    work_q.put(message["item"])
                elif message.get("op") in {"close", "bye"}:
                    work_q.put(None)
                    return
        except EOFError:
            work_q.put(None)
        finally:
            listener.close()

    threading.Thread(target=_accept_work, name=f"va-vlm-{cfg.worker_index}-work", daemon=True).start()
    # Monkeypatch via a thin wrapper: send the example once the worker has warmed up.
    _install_ready_hook(on_ready=_on_ready)
    try:
        run_vlm_worker(
            cfg,
            device=device,
            worker_index=cfg.worker_index,
            num_vlm=len(cfg.vlm_devices),
            work_q=work_q,
            prefix_q=PrefixSender(duplex),
            result_q=queue.Queue(),
            pool_q=None,
            credit_q=RpcCredit(duplex, max_batch_size=cfg.max_batch_size, timeout_s=cfg.timeout_s),
        )
    finally:
        duplex.close()


def _ae_peer(
    duplex: Duplex,
    hub: CreditHub,
    prefix_q: queue.Queue[Any],
    example: dict[str, Any],
    example_ready: threading.Event,
    cfg: RunConfig,
) -> None:
    try:
        while True:
            try:
                message = duplex.recv(timeout=cfg.timeout_s)
            except queue.Empty:
                continue
            if not isinstance(message, dict):
                continue
            op = message.get("op")
            if op == "pull":
                lane_ids = hub.acquire(int(message.get("n", 1)), timeout_s=cfg.timeout_s)
                duplex.send({"op": "credit", "ids": lane_ids})
            elif op == "release":
                hub.release([int(lane_id) for lane_id in message.get("ids", [])])
            elif op == "example":
                example["row"] = message["row"]
                example_ready.set()
            elif op == "prefix":
                prefix_q.put(_prefix_from_wire(message["msg"]))
            elif op in {"close", "bye"}:
                prefix_q.put(None)
                return
    except EOFError:
        prefix_q.put(None)
    finally:
        duplex.close()


def _prefix_to_wire(message: PrefixMsg) -> dict[str, Any]:
    return {
        "request_id": message.request_id,
        "lane_id": message.lane_id,
        "num_steps": message.num_steps,
        "scheduled_at_s": message.scheduled_at_s,
        "enqueue_ns": message.enqueue_ns,
        "vlm_done_ns": message.vlm_done_ns,
        "vlm_batch_size": message.vlm_batch_size,
        "vlm_batch_start_ns": message.vlm_batch_start_ns,
        "vlm_compute_done_ns": message.vlm_compute_done_ns,
        "vlm_export_done_ns": message.vlm_export_done_ns,
        "payload": message.payload,
    }


def _prefix_from_wire(message: dict[str, Any]) -> PrefixMsg:
    return PrefixMsg(
        request_id=str(message["request_id"]),
        lane_id=int(message["lane_id"]),
        num_steps=int(message["num_steps"]),
        scheduled_at_s=float(message["scheduled_at_s"]),
        enqueue_ns=int(message["enqueue_ns"]),
        vlm_done_ns=int(message["vlm_done_ns"]),
        vlm_batch_size=int(message["vlm_batch_size"]),
        vlm_batch_start_ns=int(message.get("vlm_batch_start_ns", 0)),
        vlm_compute_done_ns=int(message.get("vlm_compute_done_ns", 0)),
        vlm_export_done_ns=int(message.get("vlm_export_done_ns", 0)),
        payload=message.get("payload"),
    )


def _split_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host or not port:
        raise ValueError(f"expected host:port, got {addr!r}")
    return host, int(port)


def _port(addr: str) -> int:
    if not addr:
        return 0
    return _split_addr(addr)[1]


class _ignore:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return True


# Ready hook is installed per thread via a context var so local tests do not share global state.
_ready_hook = threading.local()


def _install_ready_hook(*, on_ready: Any) -> None:
    _ready_hook.fn = on_ready


def consume_ready_hook() -> Any:
    fn = getattr(_ready_hook, "fn", None)
    _ready_hook.fn = None
    return fn
