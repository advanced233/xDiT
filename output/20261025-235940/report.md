# 阶段 1 实验回报

```json
{
  "commit": "dad3342ed4c84fd4b6f26b68a3948d762078399c",
  "status": "blocked_comparison",
  "completed": [
    "int8_linear",
    "bf16 (existing)",
    "int8_both (existing)",
    "bf16_compile",
    "int8_compile",
    "int8_high",
    "int8_low"
  ],
  "errors": [
    {
      "stage": "compare_int8_both",
      "exit_code": 1,
      "log": "logs/compare_int8_both.log"
    }
  ],
  "skipped": [],
  "scope": "dad3342 section 5: single-layer gate -> compile BF16/INT8 -> eager expert quality -> comparisons/VBench",
  "linear_gate": {
    "passed": true,
    "threshold": 1.2,
    "rows": [
      {
        "variant": "torchao_compile",
        "tokens": 14040,
        "in_features": 5120,
        "out_features": 5120,
        "status": "ok",
        "ms": 2.3600128173828123,
        "samples_ms": [
          2.1549055099487306,
          2.3600128173828123,
          2.411417579650879
        ],
        "tops": 311.9052348267814,
        "speedup_vs_bf16": 1.3209528221494347,
        "relative_l2": 0.00988313090056181,
        "cosine": 0.999951183795929,
        "max_abs": 0.03515625
      },
      {
        "variant": "torchao_compile",
        "tokens": 14040,
        "in_features": 5120,
        "out_features": 13824,
        "status": "ok",
        "ms": 6.164275360107422,
        "samples_ms": [
          5.988966369628907,
          6.222540664672851,
          6.164275360107422
        ],
        "tops": 322.4176134736079,
        "speedup_vs_bf16": 1.3599787069694573,
        "relative_l2": 0.009885099716484547,
        "cosine": 0.9999512434005737,
        "max_abs": 0.03515625
      },
      {
        "variant": "torchao_compile",
        "tokens": 14040,
        "in_features": 13824,
        "out_features": 5120,
        "status": "ok",
        "ms": 5.906022262573242,
        "samples_ms": [
          5.906022262573242,
          5.9504638671875,
          5.900595092773438
        ],
        "tops": 336.51599368236424,
        "speedup_vs_bf16": 1.414695881737855,
        "relative_l2": 0.010358532890677452,
        "cosine": 0.9999464750289917,
        "max_abs": 0.037109375
      }
    ],
    "all_nn_tn_equal": true
  },
  "current": "compare_int8_both",
  "video_sources": {
    "bf16": "$XDIT_WORK/results/phase1/bf16/videos/480p_33f",
    "int8_both": "$XDIT_WORK/results/phase1/int8_both/videos/480p_33f",
    "int8_high": "$XDIT_WORK/results/phase1/int8_high/videos/480p_33f",
    "int8_low": "$XDIT_WORK/results/phase1/int8_low/videos/480p_33f"
  },
  "foreign_process_runs": 0,
  "finished": "2026-10-25T23:09:19+0800"
}
```

完整报错见 logs；视频留在服务器，不包含在压缩包中。未修改仓库代码。
