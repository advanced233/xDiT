"""Unit tests for the benchmark timing and GPU snapshot helpers.

CUDA events and nvidia-smi are replaced by a fake clock and a fake process runner,
so the step grouping and the shared-GPU checks are exercised on CPU.

Run with:
    python -m pytest tests/unit/core/utils/test_measurement.py -v
"""

import subprocess
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.core.utils.measurement import (  # noqa: E402
    COMPUTE_APP_QUERY_FIELDS,
    GPU_QUERY_FIELDS,
    ComponentCallTimer,
    gpu_snapshot,
    parse_nvidia_smi_csv,
)


class _Clock:
    def __init__(self):
        self.now = 0.0


class _Event:
    def __init__(self, clock):
        self._clock = clock
        self.time = None

    def record(self):
        self.time = self._clock.now

    def elapsed_time(self, end):
        return end.time - self.time


class _Transformer(torch.nn.Module):
    """Advances the fake clock by a fixed cost per call."""

    def __init__(self, clock, cost_ms):
        super().__init__()
        self._clock = clock
        self._cost_ms = cost_ms

    def forward(self, hidden_states, timestep=None):
        self._clock.now += self._cost_ms
        return hidden_states


def _timer(clock, components, **kwargs):
    return ComponentCallTimer(
        components,
        event_factory=lambda: _Event(clock),
        synchronize=lambda: None,
        **kwargs,
    )


def test_cfg_pairs_group_into_steps_per_expert():
    clock = _Clock()
    high, low = _Transformer(clock, 10.0), _Transformer(clock, 4.0)
    timer = _timer(clock, {"transformer": high, "transformer_2": low})
    x = torch.zeros(1)

    for t in (999.0, 950.0, 900.0):
        high(x, timestep=torch.tensor([t]))  # conditional
        high(x, timestep=torch.tensor([t]))  # unconditional
    for t in (500.0, 100.0):
        low(x, timestep=torch.tensor([t]))
        low(x, timestep=torch.tensor([t]))

    steps = timer.steps()
    assert [(step["component"], step["timestep"], step["calls"], step["ms"]) for step in steps] == [
        ("transformer", 999.0, 2, 20.0),
        ("transformer", 950.0, 2, 20.0),
        ("transformer", 900.0, 2, 20.0),
        ("transformer_2", 500.0, 2, 8.0),
        ("transformer_2", 100.0, 2, 8.0),
    ]
    summary = timer.summary()
    assert summary["total_ms"] == 76.0
    assert summary["components"]["transformer"]["steps"] == 3
    assert summary["components"]["transformer"]["calls"] == 6
    assert summary["components"]["transformer"]["step_ms_median"] == 20.0
    assert summary["components"]["transformer_2"]["total_ms"] == 16.0


def test_positional_timestep_and_a_component_that_never_ran():
    clock = _Clock()
    used, unused = _Transformer(clock, 3.0), _Transformer(clock, 1.0)
    timer = _timer(clock, {"transformer": used, "transformer_2": unused})

    used(torch.zeros(1), torch.tensor([7.0]))

    assert timer.calls() == [{"component": "transformer", "timestep": 7.0, "ms": 3.0}]
    assert timer.summary()["components"]["transformer_2"] == {"steps": 0, "calls": 0, "total_ms": 0.0}


def test_disabled_timer_records_nothing_and_reset_clears_and_detach_unhooks():
    clock = _Clock()
    module = _Transformer(clock, 2.0)
    timer = _timer(clock, {"transformer": module})
    x = torch.zeros(1)

    timer.enabled = False
    module(x, timestep=1.0)
    assert timer.calls() == []

    timer.enabled = True
    module(x, timestep=1.0)
    assert len(timer.calls()) == 1
    timer.reset()
    assert timer.calls() == []

    timer.detach()
    module(x, timestep=1.0)
    assert timer.calls() == []


def test_peak_memory_is_split_at_the_first_timed_call():
    clock = _Clock()
    module = _Transformer(clock, 1.0)
    memory = SimpleNamespace(peak=50, resets=0)

    def reset():
        memory.resets += 1
        memory.peak = 0

    timer = _timer(
        clock,
        {"transformer": module},
        track_peak_memory=True,
        max_memory_allocated=lambda: memory.peak,
        reset_peak_memory_stats=reset,
    )
    x = torch.zeros(1)

    module(x, timestep=1.0)  # the pre-call peak (50) is read, then the counter resets
    memory.peak = 30
    module(x, timestep=2.0)

    summary = timer.summary()
    assert memory.resets == 1
    assert summary["pre_call_peak_bytes"] == 50
    assert summary["call_peak_bytes"] == 30


def test_nvidia_smi_csv_numbers_and_missing_values():
    text = "0, GPU-aaa, NVIDIA A100 80GB PCIe, 15, 81920, 0, 1410, 1410, [N/A], 35\n\n"

    (row,) = parse_nvidia_smi_csv(text, GPU_QUERY_FIELDS)

    assert row["index"] == 0
    assert row["uuid"] == "GPU-aaa"
    assert row["name"] == "NVIDIA A100 80GB PCIe"
    assert row["memory.total"] == 81920
    assert row["power.draw"] is None


def test_nvidia_smi_csv_rejects_a_column_mismatch():
    with pytest.raises(ValueError, match="expected 3 nvidia-smi columns"):
        parse_nvidia_smi_csv("GPU-aaa, 1\n", COMPUTE_APP_QUERY_FIELDS)


def _fake_nvidia_smi(gpu_rows, app_rows):
    def run(command, **kwargs):
        rows = gpu_rows if command[1].startswith("--query-gpu=") else app_rows
        return SimpleNamespace(stdout="\n".join(rows))

    return run


def test_snapshot_keeps_the_requested_gpus_and_flags_other_users_processes():
    run = _fake_nvidia_smi(
        [
            "1, GPU-one, NVIDIA A100 80GB PCIe, 30000, 81920, 99, 1410, 1410, 250.5, 60",
            "2, GPU-two, NVIDIA A100 80GB PCIe, 70000, 81920, 100, 1410, 1410, 290.0, 70",
        ],
        ["GPU-one, 4242, 29000", "GPU-one, 777, 1000", "GPU-two, 888, 70000"],
    )

    snapshot = gpu_snapshot(uuids=["GPU-one"], own_pids=[4242], run=run)

    assert [gpu["index"] for gpu in snapshot["gpus"]] == [1]
    assert [process["pid"] for process in snapshot["processes"]] == [4242, 777]
    assert [process["pid"] for process in snapshot["foreign_processes"]] == [777]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("nvidia-smi"), subprocess.CalledProcessError(9, "nvidia-smi")],
)
def test_snapshot_reports_a_missing_or_failing_nvidia_smi(error):
    def run(command, **kwargs):
        raise error

    snapshot = gpu_snapshot(run=run)

    assert set(snapshot) == {"error"}
    assert type(error).__name__ in snapshot["error"]
