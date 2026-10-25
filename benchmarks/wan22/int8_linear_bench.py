"""Time Wan 14B block linears in BF16 and in torchao W8A8 INT8, eager and compiled.

    python benchmarks/wan22/int8_linear_bench.py --json $XDIT_WORK/results/int8_linear.json

Defaults are the three linear shapes of a Wan 14B block (hidden 5120, FFN 13824)
at 480p with 33 frames (14040 tokens); pass ``--tokens 14040 32760`` to add 81
frames. The INT8 layer is built by the model's own ``quantize_linear_layers_to_int8``,
so ``torchao_eager`` is exactly what ``--gemm_quantization int8`` runs without
``--use_torch_compile``, and ``torchao_compile`` is the same layer under
``torch.compile``.

Two bare ``torch._int_mm`` rows isolate the INT8 GEMM from quantization. ``int_mm_nn``
passes the weight as a contiguous (K, N) copy, the layout torchao's eager
``safe_int_mm`` hands cuBLAS; ``int_mm_tn`` passes the transposed view of the (N, K)
weight, the "TN" layout cuBLASLt requires for INT8 tensor cores. The two must give
identical int32 results.
"""

import argparse
import copy
import json

import torch
import torch.nn.functional as F
from attention_bench import _compare, _time_ms

from xfuser.core.utils.measurement import gpu_snapshot
from xfuser.core.utils.runner_utils import quantize_linear_layers_to_int8

VARIANTS = ("bf16", "torchao_eager", "torchao_compile", "int_mm_nn", "int_mm_tn")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokens", type=int, nargs="+", default=[14040], help="Rows M of the activation")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["5120x5120", "5120x13824", "13824x5120"],
        help="in_features x out_features (K x N): attention projections, FFN up, FFN down",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="Write the results to this file")
    return parser.parse_args()


def _device_snapshot():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(getattr(props, "uuid", ""))
    uuid = uuid if not uuid or uuid.startswith("GPU-") else f"GPU-{uuid}"
    snapshot = gpu_snapshot(uuids=[uuid] if uuid else None)
    gpus = snapshot.get("gpus") or [{}]
    print(f"GPU: sm clock {gpus[0].get('clocks.sm')} MHz, {gpus[0].get('temperature.gpu')} C", flush=True)
    return snapshot


def _calls(variants, tokens, linear, device, generator):
    """One zero-argument callable per variant, plus the BF16 reference output."""
    in_features, out_features = linear.in_features, linear.out_features
    x = torch.randn(1, tokens, in_features, device=device, dtype=torch.bfloat16, generator=generator)
    calls = {"bf16": lambda: F.linear(x, linear.weight, linear.bias)}
    if {"torchao_eager", "torchao_compile"} & set(variants):
        quantized = torch.nn.Sequential(copy.deepcopy(linear))
        quantize_linear_layers_to_int8(quantized)
        compiled = torch.compile(quantized, dynamic=False)
        calls["torchao_eager"] = lambda: quantized(x)
        calls["torchao_compile"] = lambda: compiled(x)
    if {"int_mm_nn", "int_mm_tn"} & set(variants):
        x_int8 = torch.randint(-128, 128, (tokens, in_features), device=device, dtype=torch.int8, generator=generator)
        w_int8 = torch.randint(
            -128, 128, (out_features, in_features), device=device, dtype=torch.int8, generator=generator
        )
        w_kn = w_int8.t().contiguous()
        calls["int_mm_nn"] = lambda: torch._int_mm(x_int8, w_kn)
        calls["int_mm_tn"] = lambda: torch._int_mm(x_int8, w_int8.t())
    return calls, calls["bf16"]()


def _measure_shape(args, tokens, in_features, out_features, device, generator):
    """Time every requested variant on one (M, K, N); tensors are freed on return."""
    linear = torch.nn.Linear(in_features, out_features, device=device, dtype=torch.bfloat16)
    calls, reference = _calls(args.variants, tokens, linear, device, generator)
    ops = 2 * tokens * in_features * out_features
    outputs, rows = {}, []
    bf16_ms = None
    for name in dict.fromkeys(["bf16", *args.variants]):
        row = {"variant": name, "tokens": tokens, "in_features": in_features, "out_features": out_features}
        try:
            outputs[name] = calls[name]()
            median_ms, samples = _time_ms(calls[name], args.warmup, args.iters, args.repeats)
        except Exception as exc:  # report and keep measuring the other variants
            rows.append({**row, "status": f"failed: {type(exc).__name__}: {exc}"})
            print(f"M={tokens:6d} {in_features:5d}x{out_features:<5d} {name:16s} {rows[-1]['status']}", flush=True)
            continue
        bf16_ms = median_ms if name == "bf16" else bf16_ms
        row.update(status="ok", ms=median_ms, samples_ms=samples, tops=ops / median_ms / 1e9)
        if bf16_ms is not None:
            row["speedup_vs_bf16"] = bf16_ms / median_ms
        if name.startswith("torchao"):
            row.update(_compare(reference, outputs[name]))
        if name == "int_mm_tn" and "int_mm_nn" in outputs:
            row["matches_int_mm_nn"] = bool(torch.equal(outputs[name], outputs["int_mm_nn"]))
        rows.append(row)
        print(
            f"M={tokens:6d} {in_features:5d}x{out_features:<5d} {name:16s} {median_ms:8.3f} ms "
            f"{row['tops']:6.1f} TOPS  x{row.get('speedup_vs_bf16', float('nan')):.2f} vs bf16"
            + (f"  rel_l2 {row['relative_l2']:.2e}" if "relative_l2" in row else "")
            + (f"  matches_nn {row['matches_int_mm_nn']}" if "matches_int_mm_nn" in row else ""),
            flush=True,
        )
    return rows


@torch.no_grad()
def main():
    args = _parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    before = _device_snapshot()
    results = []
    for tokens in args.tokens:
        for shape in args.shapes:
            in_features, out_features = (int(value) for value in shape.lower().split("x"))
            results += _measure_shape(args, tokens, in_features, out_features, device, generator)
            torch.cuda.empty_cache()
    after = _device_snapshot()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "args": vars(args),
                    "gpu_before": before,
                    "gpu_after": after,
                    "results": results,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
