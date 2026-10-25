"""Load a Wan runner once and time many generation cases with it.

    python benchmarks/wan22/bench.py --config benchmarks/wan22/configs/speed_sweep.yaml \
        --output-dir $XDIT_WORK/results/bf16

    python benchmarks/wan22/bench.py --config benchmarks/wan22/configs/speed_sweep.yaml \
        --runner-arg gemm_quantization=int8 --runner-arg int8_gemm_modules=transformer.blocks \
        --output-dir $XDIT_WORK/results/int8_high

Loading Wan 2.2 A14B takes minutes, so one process loads it once and runs every
case, prompt, seed and repeat of a config. A change to how the model is loaded
(GEMM quantization, attention backend, compile, parallel degrees) is a separate
invocation. Multi-GPU runs go through torchrun the same way; only the last rank,
the one xDiT takes the output from, writes files.

Each timed run records:

- the pipeline time (text encoding, denoising, VAE decode), from the runner;
- per-expert DiT time per denoising step, from CUDA events around every
  transformer call (``ComponentCallTimer``);
- memory: resident weights after load, the peak before the first DiT call
  (text encoding), the peak across DiT calls, and the peak of the whole run;
- ``nvidia-smi`` state of this rank's GPU before and after, with any process
  that is not this run flagged, so a timing taken next to another job can be
  dropped.

Outputs in ``--output-dir``: ``env.json``, ``runs.jsonl`` (one line per timed
run), ``summary.csv`` (medians over repeats), and videos under ``videos/``.

``--nsys-capture`` brackets only the timed runs with cudaProfilerStart/Stop, for
``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``.
"""

import argparse
import csv
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.utils.measurement import ComponentCallTimer, gpu_snapshot
from xfuser.core.utils.runner_utils import is_last_process
from xfuser.runner import xFuserModelRunner

GIB = 1024**3
CASE_KEYS = ("height", "width", "num_frames", "num_inference_steps", "guidance_scale", "guidance_scale_2", "flow_shift")
VERSION_MODULES = ("diffusers", "transformers", "torchao", "sageattention", "flash_attn", "yunchang", "xfuser")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="YAML benchmark config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runner-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override or add one runner_args entry; repeatable. VALUE is parsed as YAML.",
    )
    parser.add_argument("--prompts-file", help="Replace the config's prompts with this file, one prompt per line")
    parser.add_argument("--case", action="append", default=[], help="Run only the named case; repeatable")
    parser.add_argument("--repeats", type=int, help="Override the config's repeats")
    parser.add_argument("--nsys-capture", action="store_true")
    return parser.parse_args()


def _case_name(case):
    return case.get("name") or "x".join(str(case.get(key, "default")) for key in ("height", "width", "num_frames"))


def _load_config(path, overrides, prompts_file=None, only_cases=(), repeats=None):
    config = yaml.safe_load(Path(path).read_text())
    if prompts_file:
        config["prompts"] = []
        config["prompts_file"] = prompts_file
    if repeats is not None:
        config["repeats"] = repeats
    cases = [{**case, "name": _case_name(case)} for case in config.get("cases") or []]
    if only_cases:
        unknown = sorted(set(only_cases) - {case["name"] for case in cases})
        if unknown:
            raise SystemExit(f"--case names not in the config: {unknown}")
        cases = [case for case in cases if case["name"] in only_cases]
    config["cases"] = cases
    runner_args = dict(config.get("runner_args") or {})
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--runner-arg expects KEY=VALUE, got {item!r}")
        runner_args[key.strip()] = yaml.safe_load(value)
    config["runner_args"] = runner_args
    if not config.get("cases"):
        raise SystemExit("config needs a non-empty 'cases' list")
    prompts = list(config.get("prompts") or [])
    if config.get("prompts_file"):
        lines = Path(config["prompts_file"]).read_text().splitlines()
        prompts += [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not prompts:
        raise SystemExit("config needs 'prompts' or 'prompts_file'")
    config["prompts"] = prompts
    config.setdefault("seeds", [42])
    config.setdefault("repeats", 3)
    config.setdefault("warmup", 1)
    config.setdefault("save_videos", "first_repeat")
    if config["save_videos"] not in {"none", "first_repeat", "all"}:
        raise SystemExit("save_videos must be none, first_repeat or all")
    return config


def _runner_argv(runner_args):
    argv = []
    for key, value in runner_args.items():
        flag = f"--{key}"
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


def _runner_namespace(runner_args, output_dir):
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser(description="xFuser Arguments"))
    namespace = vars(parser.parse_args(_runner_argv(runner_args)))
    # The harness owns repetition and warmup; the runner times a single call.
    namespace.update(num_iterations=1, warmup_calls=0, output_directory=str(output_dir / "runner"))
    return namespace


def _case_args(base, case, prompt, seed):
    args = dict(base)
    args.update({key: case[key] for key in CASE_KEYS if key in case})
    args.update(prompt=[prompt], seed=seed)
    return args


def _device_uuid():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(getattr(props, "uuid", ""))
    return uuid if not uuid or uuid.startswith("GPU-") else f"GPU-{uuid}"


def _versions():
    versions = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    versions["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    for name in VERSION_MODULES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # optional packages may be absent or fail to load
            versions[name] = f"unavailable ({type(exc).__name__})"
    return versions


def _git_state():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _environment(config, namespace):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    driver = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id={_device_uuid()}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    runner_view = {key: value for key, value in namespace.items() if key not in {"prompt", "negative_prompt"}}
    return {
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": _versions(),
        "git": _git_state(),
        "device": {
            "name": props.name,
            "uuid": _device_uuid(),
            "capability": f"{props.major}.{props.minor}",
            "total_memory_gib": props.total_memory / GIB,
            "driver": driver,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "config": {key: value for key, value in config.items() if key != "prompts"},
        "prompts": config["prompts"],
        "runner_args": runner_view,
    }


def _components(model):
    names = getattr(model.settings, "transformer_attr_names", None) or ["transformer"]
    return {name: getattr(model.pipe, name) for name in names if getattr(model.pipe, name, None) is not None}


def _save_video(model, output, path):
    from diffusers.utils import export_to_video

    (video, _), *_ = list(output.get_outputs())
    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(path), fps=model.settings.fps)
    return str(path)


def _gib(value):
    return None if value is None else value / GIB


def _summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["case"], record["prompt_index"], record["seed"]), []).append(record)
    rows = []
    for (case, prompt_index, seed), runs in groups.items():
        clean = [run for run in runs if not run["foreign_processes"]] or runs
        row = {
            "case": case,
            "prompt_index": prompt_index,
            "seed": seed,
            "repeats": len(runs),
            "clean_repeats": sum(1 for run in runs if not run["foreign_processes"]),
            "pipeline_s_median": statistics.median(run["pipeline_s"] for run in clean),
            "dit_ms_median": statistics.median(run["dit"]["total_ms"] for run in clean),
            "dit_peak_gib_max": max((run["memory"]["dit_peak_gib"] or 0.0) for run in clean),
            "run_peak_gib_max": max((run["memory"]["run_peak_gib"] or 0.0) for run in clean),
        }
        for name in clean[0]["dit"]["components"]:
            stats = [run["dit"]["components"][name] for run in clean]
            row[f"{name}_steps"] = stats[0]["steps"]
            row[f"{name}_total_ms_median"] = statistics.median(stat["total_ms"] for stat in stats)
            if stats[0]["steps"]:
                row[f"{name}_step_ms_median"] = statistics.median(stat["step_ms_median"] for stat in stats)
        rows.append(row)
    return rows


def _write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = _parse_args()
    output_dir = Path(cli.output_dir)
    config = _load_config(cli.config, cli.runner_arg, cli.prompts_file, cli.case, cli.repeats)
    runs_path = output_dir / "runs.jsonl"
    if runs_path.exists():
        raise SystemExit(f"{runs_path} already exists; pick a new --output-dir so runs are not mixed")
    namespace = _runner_namespace(config["runner_args"], output_dir)
    writer = is_last_process()
    if writer:
        output_dir.mkdir(parents=True, exist_ok=True)

    runner = xFuserModelRunner(namespace)
    first_case = config["cases"][0]
    runner.initialize(
        runner.preprocess_args(_case_args(namespace, first_case, config["prompts"][0], config["seeds"][0]))
    )
    model = runner.model
    torch.cuda.synchronize()
    load = {
        "resident_gib": _gib(torch.cuda.memory_allocated()),
        "load_peak_gib": _gib(torch.cuda.max_memory_allocated()),
    }
    if writer:
        environment = _environment(config, namespace)
        environment["load"] = load
        (output_dir / "env.json").write_text(json.dumps(environment, indent=2, default=str))

    timer = ComponentCallTimer(_components(model), track_peak_memory=True, nvtx=cli.nsys_capture)
    uuid = _device_uuid()
    records = []
    for case in config["cases"]:
        name = case["name"]
        timer.enabled = False
        warmup_args = runner.preprocess_args(_case_args(namespace, case, config["prompts"][0], config["seeds"][0]))
        for _ in range(int(config["warmup"])):
            runner.run(warmup_args)
        timer.enabled = True

        for prompt_index, prompt in enumerate(config["prompts"]):
            for seed in config["seeds"]:
                input_args = runner.preprocess_args(_case_args(namespace, case, prompt, seed))
                for repeat in range(int(config["repeats"])):
                    before = gpu_snapshot(uuids=[uuid], own_pids=[os.getpid()])
                    timer.reset()
                    torch.cuda.reset_peak_memory_stats()
                    if cli.nsys_capture:
                        torch.cuda.profiler.start()
                    output, timings = runner.run(input_args)
                    torch.cuda.synchronize()
                    if cli.nsys_capture:
                        torch.cuda.profiler.stop()
                    dit = timer.summary()
                    # The timer restarts the peak counter at the first DiT call, so the counter now
                    # covers DiT plus VAE decode; the text-encoding peak before it comes from the timer.
                    run_peak = max(torch.cuda.max_memory_allocated(), dit.get("pre_call_peak_bytes") or 0)
                    after = gpu_snapshot(uuids=[uuid], own_pids=[os.getpid()])
                    foreign = before.get("foreign_processes", []) + after.get("foreign_processes", [])
                    record = {
                        "case": name,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "repeat": repeat,
                        **{key: input_args.get(key) for key in CASE_KEYS},
                        "pipeline_s": timings[-1],
                        "dit": {key: value for key, value in dit.items() if not key.endswith("_bytes")},
                        "memory": {
                            **load,
                            "pre_dit_peak_gib": _gib(dit.get("pre_call_peak_bytes")),
                            "dit_peak_gib": _gib(dit.get("call_peak_bytes")),
                            "run_peak_gib": _gib(run_peak),
                        },
                        "gpu_before": before,
                        "gpu_after": after,
                        "foreign_processes": foreign,
                        "video": None,
                    }
                    save = config["save_videos"] == "all" or (config["save_videos"] == "first_repeat" and repeat == 0)
                    if writer and save:
                        video = output_dir / "videos" / name / f"p{prompt_index:03d}_s{seed}.mp4"
                        if config["save_videos"] == "all":
                            video = video.with_name(f"p{prompt_index:03d}_s{seed}_r{repeat}.mp4")
                        record["video"] = _save_video(model, output, video)
                    records.append(record)
                    if writer:
                        with runs_path.open("a") as handle:
                            handle.write(json.dumps(record, default=str) + "\n")
                    print(
                        f"[bench] {name} p{prompt_index} s{seed} r{repeat}: pipeline {record['pipeline_s']:.2f}s, "
                        f"DiT {dit['total_ms'] / 1000:.2f}s" + (" (other processes on this GPU)" if foreign else ""),
                        flush=True,
                    )

    timer.detach()
    if writer:
        _write_csv(output_dir / "summary.csv", _summarize(records))
        print(f"[bench] results in {output_dir}", flush=True)
    runner.cleanup()


if __name__ == "__main__":
    main()
