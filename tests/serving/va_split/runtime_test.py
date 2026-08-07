# ruff: noqa: SLF001

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split import runtime as runtime_module
from openpi.serving.va_split.runtime import LocalVASplitRuntime
from openpi.serving.va_split.runtime import ProcessVASplitRuntime
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import Shutdown


class FakeSplitModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.prefix_batch_sizes: list[int] = []

    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
        self.prefix_batch_sizes.append(int(observation.state.shape[0]))
        return PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(observation.state.shape[0], 3, dtype=torch.bool),
            state=observation.state,
        )

    def init_denoise_state(
        self, device: str, batch_size: int, noise: torch.Tensor | None, num_steps: int
    ) -> DenoiseState:
        assert device == "cpu"
        if noise is None:
            noise = torch.zeros(batch_size, self.config.action_horizon, self.config.action_dim)
        return DenoiseState(
            x_t=noise,
            step_idx=torch.tensor(0, dtype=torch.int32),
            num_steps=num_steps,
            dt=torch.tensor(-1.0 / num_steps),
        )

    def denoise_one_batch(self, prefix_batch: PrefixFeature, denoise_batch: DenoiseState) -> torch.Tensor:
        assert prefix_batch.prefix_pad_masks.shape[0] == denoise_batch.x_t.shape[0]
        return torch.ones_like(denoise_batch.x_t)


def test_local_va_split_runtime_runs_vlm_then_ae_and_releases_prefix():
    runtime = LocalVASplitRuntime(model=FakeSplitModel(), device="cpu", max_ae_batch_size=2)
    image = torch.zeros(1, 3, 224, 224)
    observation = {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image.clone(),
            "right_wrist_0_rgb": image.clone(),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        "state": torch.zeros(1, 2),
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    result = runtime.infer(observation, {"num_steps": 1, "noise": torch.zeros(1, 2, 1)})

    assert result.request_id
    torch.testing.assert_close(result.actions, -torch.ones(1, 2, 1))
    assert runtime.vlm_worker.live_features == {}


def test_local_va_split_runtime_infer_batch_builds_prefix_once_and_returns_row_order():
    model = FakeSplitModel()
    runtime = LocalVASplitRuntime(model=model, device="cpu", max_ae_batch_size=2)
    image = torch.zeros(2, 3, 224, 224)
    observation = {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image.clone(),
            "right_wrist_0_rgb": image.clone(),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        },
        "state": torch.zeros(2, 2),
        "tokenized_prompt": torch.ones(2, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    result = runtime.infer_batch(observation, {"num_steps": 1, "noise": torch.zeros(2, 2, 1)})

    assert result.request_id
    assert model.prefix_batch_sizes == [2]
    torch.testing.assert_close(result.actions, -torch.ones(2, 2, 1))
    assert result.timing is not None
    assert result.timing["effective_batch"] == 2
    assert result.timing["vlm_effective_batch"] == 2.0
    assert result.timing["ae_effective_batch_mean"] == 2.0
    assert runtime.vlm_worker.live_features == {}


class _ResultQueue:
    def __init__(self, messages, *, get_delay_s: float = 0.0):
        self._messages = list(messages)
        self._get_delay_s = get_delay_s

    def get(self):
        if self._get_delay_s:
            time.sleep(self._get_delay_s)
        return self._messages.pop(0)


def test_process_runtime_collect_results_records_ae_result_transfer_latency():
    enqueue_ns = time.monotonic_ns() - 5_000_000
    runtime = object.__new__(ProcessVASplitRuntime)
    runtime._result_queue = _ResultQueue(
        [
            ActionResult(
                request_id="req-1",
                actions=torch.zeros(1, 2, 1),
                timing={"_ae_result_enqueue_ns": float(enqueue_ns)},
            ),
            Shutdown(),
        ],
        get_delay_s=0.01,
    )
    runtime._condition = threading.Condition()
    runtime._pending_results = {}
    runtime._pending_errors = {}
    runtime._shutdown_seen = False

    runtime._collect_results()

    timing = runtime._pending_results["req-1"].timing
    assert timing is not None
    assert timing["ae_result_queue_wait_ms"] >= 4.0
    assert timing["ae_result_transfer_ms"] >= 8.0
    assert timing["va_split_transfer_ms"] == timing["ae_result_transfer_ms"]
    assert timing["va_split_queue_wait_ms"] == timing["ae_result_queue_wait_ms"]
    assert "_ae_result_enqueue_ns" not in timing


def test_process_runtime_waits_for_both_workers_before_result_thread(monkeypatch):
    events: list[str] = []
    processes: dict[str, object] = {}

    class FakeQueue:
        def put(self, value):
            events.append(f"queue-put:{type(value).__name__}")

        def get(self):
            raise AssertionError("ready queue should be intercepted by the test helper")

    class FakeProcess:
        def __init__(self, *, target, args, daemon):
            del args
            self.role = "vlm" if target is runtime_module._run_vlm_process else "ae"
            self.daemon = daemon
            self.started = False
            processes[self.role] = self

        def start(self):
            self.started = True
            events.append(f"start:{self.role}")

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return self.started

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, *, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    class FakeThread:
        def __init__(self, *, target, daemon):
            del target
            self.daemon = daemon

        def start(self):
            events.append("thread:start")

        def join(self, timeout=None):
            del timeout

    def collect_ready(_queue, *, expected_roles, timeout_s, vlm_process, ae_process):
        del timeout_s
        events.append("ready:" + ",".join(expected_roles))
        assert expected_roles == ("vlm", "ae")
        assert vlm_process is processes["vlm"]
        assert ae_process is processes["ae"]
        assert processes["vlm"].started
        assert processes["ae"].started

    monkeypatch.setattr(runtime_module.torch.multiprocessing, "get_context", lambda _start_method: FakeContext())
    monkeypatch.setattr(runtime_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(runtime_module, "_collect_worker_ready", collect_ready, raising=False)

    runtime = ProcessVASplitRuntime(
        model_factory=object(),
        device="cpu",
        start_method="spawn",
    )

    assert events.index("ready:vlm,ae") > max(events.index("start:vlm"), events.index("start:ae"))
    assert events[-1] == "thread:start"
    assert runtime._vlm_process is processes["vlm"]
    assert runtime._ae_process is processes["ae"]

def test_process_runtime_passes_role_specific_model_factories(monkeypatch):
    process_args = []

    class FakeQueue:
        def put(self, value):
            del value

        def get(self):
            return Shutdown()

    class FakeProcess:
        def __init__(self, *, target, args, daemon):
            del target
            self.args = args
            self.daemon = daemon
            process_args.append(args)

        def start(self):
            pass

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return False

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, *, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    class FakeThread:
        def __init__(self, *, target, daemon):
            del target
            self.daemon = daemon

        def start(self):
            pass

        def join(self, timeout=None):
            del timeout

    base_factory = object()
    vlm_factory = object()
    ae_factory = object()
    monkeypatch.setattr(runtime_module.torch.multiprocessing, "get_context", lambda _start_method: FakeContext())
    monkeypatch.setattr(runtime_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(runtime_module, "_collect_worker_ready", lambda *args, **kwargs: None, raising=False)

    runtime = ProcessVASplitRuntime(
        model_factory=base_factory,
        vlm_model_factory=vlm_factory,
        ae_model_factory=ae_factory,
        device="cpu",
    )

    assert runtime._model_factory is base_factory
    assert runtime._vlm_model_factory is vlm_factory
    assert runtime._ae_model_factory is ae_factory
    assert process_args[0][0] is vlm_factory
    assert process_args[1][0] is ae_factory
