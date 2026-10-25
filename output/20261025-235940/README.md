# 阶段 1 实验结果（部分完成）

实验提交：`dad3342ed4c84fd4b6f26b68a3948d762078399c`。原有 BF16、INT8 双专家 eager 结果来自 `61e8a20`，各组 `env.json` 保留其实际提交信息。

- 5.1 单层计时已完成：14040 tokens 下三种形状的 compile 加速比为 1.321×、1.360×、1.415×；全部 NN/TN 整数结果一致。
- 5.2 BF16 compile 与 INT8 compile 两组均完成，各有 33 帧及 81 帧结果。
- 5.3 INT8 高噪声、低噪声 eager 两组均完成，各 10 个视频；连同原有两组，共生成 40 个视频。
- 视频对比未完成：LPIPS 首次下载 AlexNet 权重时发生 `BrokenPipeError`，完整报错在 `logs/compare_int8_both.log`。
- VBench 尚未执行；没有可报告的 LPIPS / PSNR / SSIM 或 VBench 分数。
- 已记录运行中同卡其他计算进程数为 0，没有因此重跑的 case。

目录保留各组 `env.json`、`summary.csv`、`runs.jsonl`、单层计时、门槛检查及完整日志。视频和模型权重不上传。本地路径、用户名、主机名及 GPU UUID 已脱敏，`$XDIT_WORK` 是占位符。

`workflow-status.json` 的 `blocked_comparison` 是最终实验状态；外层调度脚本正常退出不代表所有实验完成。此前的取消及调度中断日志也保留，不能将用户主动取消当作模型运行失败。

81 帧 compile 的 pipeline 总时间包含新形状编译开销，应按手册比较两个专家的单步中位数，不能直接拿该总时间宣称推理加速。
