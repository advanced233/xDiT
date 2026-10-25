# 阶段 1：剩余评测结果

实验版本 `3038c24fd7d4b1783eb37578421799bd152a0c41`。最新状态见 `remaining-status.json`，完整执行记录见 `remaining-report.md`。旧 `workflow-status.json` / `report.md` 是此前在 LPIPS 下载处停止时的历史记录。

本次评测使用既有四组视频，未重跑生成。AlexNet 校验、各维度下载时限和失败原因均已记录；未产生的指标不能当作通过。各组 VBench 输出在 `vbench_*`，对比在 `compare_*_vs_bf16.json`，注意力测试在 `attention.json`（若对应步骤成功）。

视频、模型权重不上传，主机名、用户名、本地工作路径及 GPU UUID 已脱敏。源码未修改。
