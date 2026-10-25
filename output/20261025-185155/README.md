# Phase 1: BF16 versus eager INT8 evidence

Experiment commit: `61e8a20a80aab36d54363f7e76ccdc9d882f8ac9`.
Single A100 PCIe 80GB, physical GPU 1. CUDNN attention. No torch.compile, no warmup, no separate smoke.

Completed: BF16 and both-expert INT8, each 10 x 33-frame / 40-step videos and one 81-frame / 10-step timing. All 22 recorded runs have no foreign compute processes. Expert steps: 26/14 at 40 steps; 7/3 at 10 steps.

INT8 high-only was briefly started during recovery and cancelled by user before results. Low-only, video comparisons and VBench are not completed. Original coordinator disappeared without a recorded exit status; cause is unknown. User requested stopping further generation and investigating slowdown.

Evidence includes per-run data, environment metadata, summary tables and full benchmark logs. Videos are retained locally, excluded here. Local work paths and GPU UUIDs are redacted; repository source files are unchanged.

`metrics.json` aggregates per-prompt medians for 33 frames and the single 81-frame run. Higher runtime means slower; speedup is BF16 time divided by INT8 time.

Initial finding: slower INT8 execution is concentrated in DiT, not model loading or video output. Thermal slowdown was observed during one INT8 sample; BF16 has only boundary snapshots, so comparative thermal attribution is not established. Source analysis will be added separately; no cause is claimed proven yet.
