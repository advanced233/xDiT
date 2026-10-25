"""Time xDiT's dense attention backends at Wan 2.x shapes, to pick the baseline.

    python benchmarks/wan22/attention_bench.py --json $XDIT_WORK/results/attention.json

Defaults cover Wan 14B self-attention (head_dim 128) at 480p with 33 and 81
frames (14040 and 32760 tokens), with all 40 heads and with the 20 a rank holds
under TP2 or Ulysses2. Every backend is called through its registered spec, the
same dispatch the model uses, so constraint checks and fallbacks are part of what
is timed. Outputs are compared with the reference backend's (relative L2 and
cosine), which matters for the low-precision SAGE path.

The fastest dense backend becomes the attention baseline for every later
experiment: a faster attention raises the GEMM share of DiT time, and with it
the speedup INT8 GEMMs can show.
"""

import argparse
import json
import statistics

import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backends", nargs="+", default=["CUDNN", "SDPA_FLASH", "FLASH", "SAGE"])
    parser.add_argument("--reference", default="SDPA_FLASH", help="Backend whose output the others are compared to")
    parser.add_argument("--heads", type=int, nargs="+", default=[40, 20])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[14040, 32760])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="Write the results to this file")
    return parser.parse_args()


def _time_ms(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples), samples


def _compare(reference, output):
    reference = reference.float()
    output = output.float()
    delta = output - reference
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(reference.flatten(), output.flatten(), dim=0)),
        "max_abs": float(delta.abs().max()),
    }


def _measure_shape(args, specs, seq_len, heads, device, dtype, generator):
    """Time every requested backend on one (seq_len, heads) shape; tensors are freed on return."""
    shape = (args.batch, heads, seq_len, args.head_dim)
    query, key, value = (torch.randn(shape, device=device, dtype=dtype, generator=generator) for _ in range(3))
    call = AttnCall()
    flops = 4 * args.batch * heads * seq_len * seq_len * args.head_dim
    reference_output = None
    ref_spec, ref_reason = specs[args.reference]
    if ref_reason is None:
        reference_output, _ = ref_spec.run(query, key, value, call)

    rows = []
    for name in args.backends:
        spec, reason = specs[name]
        row = {"backend": name, "seq_len": seq_len, "heads": heads, "head_dim": args.head_dim, "dtype": args.dtype}
        if reason is not None:
            rows.append({**row, "status": f"unavailable: {reason}"})
            continue
        try:
            output, _ = spec.run(query, key, value, call)
            median_ms, samples = _time_ms(
                lambda: spec.run(query, key, value, call), args.warmup, args.iters, args.repeats
            )
        except Exception as exc:  # report and keep measuring the other backends
            rows.append({**row, "status": f"failed: {type(exc).__name__}: {exc}"})
            continue
        row.update(status="ok", ms=median_ms, samples_ms=samples, tflops=flops / median_ms / 1e9)
        if reference_output is not None:
            row.update(_compare(reference_output, output))
        rows.append(row)
        print(
            f"L={seq_len:6d} H={heads:3d} {name:12s} {median_ms:8.2f} ms {row['tflops']:7.1f} TFLOPS"
            + (f"  rel_l2 vs {args.reference} {row['relative_l2']:.2e}" if "relative_l2" in row else ""),
            flush=True,
        )
    return rows


def main():
    args = _parse_args()
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    specs = {}
    for name in dict.fromkeys([args.reference, *args.backends]):
        spec = registry.find(AttentionBackendType[name])
        reason = "no registered spec" if spec is None else spec.unavailable()
        specs[name] = (spec, reason)
        if reason is not None:
            print(f"{name}: unavailable ({reason})")

    results = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for seq_len in args.seq_lens:
        for heads in args.heads:
            results += _measure_shape(args, specs, seq_len, heads, device, dtype, generator)
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "args": vars(args),
                    "results": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
