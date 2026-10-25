"""Compare candidate videos with same-seed reference videos: LPIPS, PSNR, SSIM.

    python benchmarks/wan22/compare_videos.py \
        --reference $XDIT_WORK/results/phase1/bf16/videos/480p_33f \
        --candidate $XDIT_WORK/results/phase1/int8_both/videos/480p_33f \
        --json $XDIT_WORK/results/phase1/int8_vs_bf16.json

Files are matched by name, which ``bench.py`` derives from prompt index and seed.
These metrics measure how far a run drifts from the reference, not how good it
looks: a change to the high-noise expert can move the layout and score badly
here while looking just as good, so read them next to VBench. Both sides are
decoded MP4s, so codec noise sets a floor; compare two BF16 runs of the same
seed to see it, which also shows whether the reference is deterministic.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True, help="Directory of reference .mp4 files")
    parser.add_argument("--candidate", required=True, help="Directory of candidate .mp4 files")
    parser.add_argument("--lpips-net", default="alex", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS (no network weights needed)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8, help="Frames per LPIPS/SSIM batch")
    parser.add_argument("--json", help="Write per-video and mean metrics to this file")
    return parser.parse_args()


def read_frames(path: Path) -> np.ndarray:
    """All frames of a video as uint8 RGB, shape (T, H, W, 3)."""
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def _to_tensor(frames: np.ndarray, device) -> torch.Tensor:
    """uint8 (T, H, W, 3) to float (T, 3, H, W) in [0, 1]."""
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR in dB for images in [0, 1]; identical frames give inf."""
    mse = (reference - candidate).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Per-frame SSIM (Wang et al. 2004, 11x11 Gaussian, sigma 1.5), averaged over RGB."""
    channels = reference.shape[1]
    window = _gaussian_window(channels=channels, device=reference.device)
    c1, c2 = 0.01**2, 0.03**2

    def blur(x):
        return F.conv2d(x, window, groups=channels)

    mu_x, mu_y = blur(reference), blur(candidate)
    sigma_x = blur(reference * reference) - mu_x**2
    sigma_y = blur(candidate * candidate) - mu_y**2
    sigma_xy = blur(reference * candidate) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return score.flatten(1).mean(dim=1)


def _finite_mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("inf")


def compare(reference_path: Path, candidate_path: Path, device, batch: int, lpips_model=None) -> dict:
    reference_frames = read_frames(reference_path)
    candidate_frames = read_frames(candidate_path)
    if reference_frames.shape != candidate_frames.shape:
        return {"error": f"shape mismatch {reference_frames.shape} vs {candidate_frames.shape}"}
    psnr_values, ssim_values, lpips_values = [], [], []
    with torch.no_grad():
        for start in range(0, len(reference_frames), batch):
            reference = _to_tensor(reference_frames[start : start + batch], device)
            candidate = _to_tensor(candidate_frames[start : start + batch], device)
            psnr_values += psnr(reference, candidate).tolist()
            ssim_values += ssim(reference, candidate).tolist()
            if lpips_model is not None:
                lpips_values += lpips_model(reference * 2 - 1, candidate * 2 - 1).flatten().tolist()
    result = {
        "frames": len(reference_frames),
        "psnr_db": _finite_mean(psnr_values),
        "identical_frames": sum(1 for value in psnr_values if not math.isfinite(value)),
        "ssim": statistics.fmean(ssim_values),
    }
    if lpips_model is not None:
        result["lpips"] = statistics.fmean(lpips_values)
    return result


def main():
    args = _parse_args()
    reference_dir, candidate_dir = Path(args.reference), Path(args.candidate)
    names = sorted(path.name for path in reference_dir.glob("*.mp4"))
    missing = [name for name in names if not (candidate_dir / name).exists()]
    pairs = [name for name in names if name not in missing]
    if not pairs:
        raise SystemExit(f"no matching .mp4 names between {reference_dir} and {candidate_dir}")

    lpips_model = None
    if not args.no_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net=args.lpips_net, verbose=False).to(args.device).eval()

    per_video = {}
    for name in pairs:
        per_video[name] = compare(reference_dir / name, candidate_dir / name, args.device, args.batch, lpips_model)
        print(f"{name}: {per_video[name]}", flush=True)

    scored = [metrics for metrics in per_video.values() if "error" not in metrics]
    mean = {
        key: _finite_mean([metrics[key] for metrics in scored])
        for key in ("psnr_db", "ssim", "lpips")
        if scored and key in scored[0]
    }
    print(f"mean over {len(scored)} videos: {mean}; unmatched in candidate: {missing}", flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "reference": str(reference_dir),
                    "candidate": str(candidate_dir),
                    "mean": mean,
                    "videos": per_video,
                    "missing_in_candidate": missing,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
