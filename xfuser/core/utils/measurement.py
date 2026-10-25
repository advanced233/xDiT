"""Per-component CUDA timing and GPU state snapshots for benchmark scripts.

Benchmarks under ``benchmarks/`` attach these to a loaded pipeline. The timer
splits denoising time by transformer, so Wan 2.2's two experts are reported
separately, and by denoising step. The snapshot records what else is running on
the GPU, so a timing taken next to another user's job on a shared machine can
be recognised and discarded.
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
    "temperature.gpu",
)
COMPUTE_APP_QUERY_FIELDS = ("gpu_uuid", "pid", "used_memory")


@dataclass
class _Call:
    component: str
    timestep: object
    start: object
    end: object = None


def _timestep_value(timestep) -> Optional[float]:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return float(timestep.reshape(-1)[0].item())
    return float(timestep)


def _cuda_event():
    return torch.cuda.Event(enable_timing=True)


class ComponentCallTimer:
    """Time every forward call of named modules with CUDA events.

    Events are recorded on the current stream when a call starts and ends, and
    read only in :meth:`calls`, after a synchronize, so timing a call adds no
    host sync to it. The timestep argument is kept as a tensor until then for
    the same reason. Consecutive calls of one component at one timestep form a
    denoising step: a pipeline that runs the conditional and unconditional
    branches separately calls its transformer twice per step.

    With ``track_peak_memory`` the allocator's peak counter is reset on the
    first call after :meth:`reset`. The peak before that call, which covers text
    encoding, and the peak across the timed calls are both reported, and the
    counter afterwards also covers whatever runs after the last call (VAE decode).

    Attach to the module the pipeline calls. Under torch.compile that is the
    compiled wrapper, so the hooks stay outside the compiled region.
    """

    def __init__(
        self,
        components: Mapping[str, torch.nn.Module],
        *,
        timestep_kwarg: str = "timestep",
        timestep_position: int = 1,
        track_peak_memory: bool = False,
        nvtx: bool = False,
        event_factory: Callable[[], object] = _cuda_event,
        synchronize: Callable[[], None] = torch.cuda.synchronize,
        max_memory_allocated: Callable[[], int] = torch.cuda.max_memory_allocated,
        reset_peak_memory_stats: Callable[[], None] = torch.cuda.reset_peak_memory_stats,
    ) -> None:
        if not components:
            raise ValueError("ComponentCallTimer needs at least one component")
        self.enabled = True
        self._timestep_kwarg = timestep_kwarg
        self._timestep_position = timestep_position
        self._track_peak_memory = track_peak_memory
        self._nvtx = nvtx
        self._event_factory = event_factory
        self._synchronize = synchronize
        self._max_memory_allocated = max_memory_allocated
        self._reset_peak_memory_stats = reset_peak_memory_stats
        self._handles = []
        self._open: Dict[str, List[_Call]] = {name: [] for name in components}
        self.reset()
        for name, module in components.items():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True))
            self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def reset(self) -> None:
        self._calls: List[_Call] = []
        for stack in self._open.values():
            stack.clear()
        self._pre_call_peak: Optional[int] = None
        self._call_peak: Optional[int] = None

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ComponentCallTimer":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    def _pre_hook(self, name: str):
        def hook(module, args, kwargs):
            if not self.enabled:
                return None
            if self._track_peak_memory and self._pre_call_peak is None:
                self._pre_call_peak = self._max_memory_allocated()
                self._reset_peak_memory_stats()
            timestep = kwargs.get(self._timestep_kwarg)
            if timestep is None and len(args) > self._timestep_position:
                timestep = args[self._timestep_position]
            if self._nvtx:
                torch.cuda.nvtx.range_push(name)
            start = self._event_factory()
            start.record()
            call = _Call(component=name, timestep=timestep, start=start)
            self._open[name].append(call)
            self._calls.append(call)
            return None

        return hook

    def _post_hook(self, name: str):
        def hook(module, args, output):
            stack = self._open[name]
            if not stack:
                return None
            end = self._event_factory()
            end.record()
            stack.pop().end = end
            if self._nvtx:
                torch.cuda.nvtx.range_pop()
            if self._track_peak_memory:
                peak = self._max_memory_allocated()
                self._call_peak = peak if self._call_peak is None else max(self._call_peak, peak)
            return None

        return hook

    def calls(self) -> List[dict]:
        """Completed calls in order, with their duration in milliseconds."""
        self._synchronize()
        return [
            {
                "component": call.component,
                "timestep": _timestep_value(call.timestep),
                "ms": float(call.start.elapsed_time(call.end)),
            }
            for call in self._calls
            if call.end is not None
        ]

    def steps(self) -> List[dict]:
        """Calls grouped into denoising steps, in order."""
        grouped: List[dict] = []
        for call in self.calls():
            last = grouped[-1] if grouped else None
            if last and last["component"] == call["component"] and last["timestep"] == call["timestep"]:
                last["calls"] += 1
                last["ms"] += call["ms"]
            else:
                grouped.append({**call, "calls": 1})
        return grouped

    def summary(self) -> dict:
        steps = self.steps()
        components = {}
        for name in self._open:
            durations = [step["ms"] for step in steps if step["component"] == name]
            if not durations:
                components[name] = {"steps": 0, "calls": 0, "total_ms": 0.0}
                continue
            components[name] = {
                "steps": len(durations),
                "calls": sum(step["calls"] for step in steps if step["component"] == name),
                "total_ms": sum(durations),
                "first_step_ms": durations[0],
                "step_ms_mean": statistics.fmean(durations),
                "step_ms_median": statistics.median(durations),
                "step_ms_min": min(durations),
                "step_ms_max": max(durations),
            }
        result = {
            "total_ms": sum(component["total_ms"] for component in components.values()),
            "components": components,
        }
        if self._track_peak_memory:
            result["pre_call_peak_bytes"] = self._pre_call_peak
            result["call_peak_bytes"] = self._call_peak
        return result


def _csv_value(raw: str):
    value = raw.strip()
    if not value or value.startswith("[") or value in {"N/A", "Not Supported"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_nvidia_smi_csv(text: str, fields: Sequence[str]) -> List[dict]:
    """Parse ``nvidia-smi --query-*=... --format=csv,noheader,nounits`` output."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split(",")
        if len(values) != len(fields):
            raise ValueError(f"expected {len(fields)} nvidia-smi columns, got {len(values)}: {line!r}")
        rows.append({field: _csv_value(value) for field, value in zip(fields, values)})
    return rows


def gpu_snapshot(
    uuids: Optional[Sequence[str]] = None,
    own_pids: Sequence[int] = (),
    run: Callable = subprocess.run,
) -> dict:
    """Query nvidia-smi for GPU state and the compute processes on those GPUs.

    ``uuids`` limits both lists to the given devices, in nvidia-smi's
    ``GPU-...`` form. A process on them whose pid is not in ``own_pids`` is
    reported under ``foreign_processes``. Inside a container with its own pid
    namespace nvidia-smi may report host pids, which then all look foreign.
    """

    def query(flag: str, fields: Sequence[str]) -> List[dict]:
        completed = run(
            ["nvidia-smi", f"{flag}={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return parse_nvidia_smi_csv(completed.stdout, fields)

    try:
        gpus = query("--query-gpu", GPU_QUERY_FIELDS)
        processes = query("--query-compute-apps", COMPUTE_APP_QUERY_FIELDS)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if uuids is not None:
        wanted = set(uuids)
        gpus = [gpu for gpu in gpus if gpu["uuid"] in wanted]
        processes = [process for process in processes if process["gpu_uuid"] in wanted]
    own = set(own_pids)
    return {
        "gpus": gpus,
        "processes": processes,
        "foreign_processes": [process for process in processes if process["pid"] not in own],
    }
