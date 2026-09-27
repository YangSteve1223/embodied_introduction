# EXP005 综合后训练与 RLinf 加速实验报告

日期：2026-09-23  
服务器：`act-server`，运行目录：`/share/yangpengju-local/openvla-oft-repro`

## 结论先行

目前最好的特化结果是 `object_lora_replay_rw8`：在固定的 LIBERO Spatial + Object held-out 640 样本上，归一化 action-chunk L1 从 0.164500 降到 0.112032，整体误差下降 31.90%；Object 从 0.277858 降到 0.171846，下降 38.15%；Spatial 从 0.051142 变为 0.052217，基本保持（下降 2.10% 的误差指标，约 2.1% 波动）。该结果对应一个 rank-8 LoRA adapter，权重位于：

`/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260923/object_lora_replay_rw8/adapter/adapter_model.safetensors`

RLinf 的多卡加速在 state-only ManiSkill/PickCube 系统基准上是成立的：四卡相对单卡的完整训练吞吐为 3.48x，rollout 交互吞吐为 3.98x，四卡效率分别约为理想 4x 的 87.1% 和 99.6%。

## VLA 后训练结果

协议：OpenVLA-OFT 官方 7B LIBERO-Spatial checkpoint；`openvla/modified_libero_rlds`；Spatial/Object 两个 suite，各任务 episode-level 固定拆分；held-out 640 transitions；指标是归一化 action-chunk L1，越低越好。

| 方法 | 参数/训练形式 | Macro L1 | Spatial | Object | 相对 baseline |
|---|---|---:|---:|---:|---:|
| Official baseline | 7B | 0.164500 | 0.051142 | 0.277858 | — |
| Head + proprio SFT | 冻结 backbone | 0.137058 | 0.065988 | 0.208129 | -16.68% |
| Head + proprio distill | teacher consistency | 0.135026 | 0.063058 | 0.206993 | -17.92% |
| Head replay rw8 | replay retention | 0.132775 | 0.076386 | 0.189165 | -19.29% |
| Object LoRA replay rw4 | rank-4 LoRA | 0.114254 | 0.053972 | 0.174536 | -30.54% |
| **Object LoRA replay rw8** | **rank-8 LoRA** | **0.112032** | **0.052217** | **0.171846** | **-31.90%** |
| Offline AWR LoRA rw8 | advantage-weighted regression | 0.121694 | 0.056318 | 0.187071 | -26.02% |

因此当前建议汇报的主结果是：Object 特化采用 LoRA + replay，能明显提高 Object 误差指标，同时不明显损伤原 Spatial 能力。由于服务器 LIBERO EGL 渲染设备不可用，以上是离线 action prediction 指标，不冒充 closed-loop success rate。

### 蒸馏结果

- 4-layer 截断学生模型约 2.04B 参数，完整权重：
  `/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260922/06_true_distill/truncated4_2b_1000/student_step_1000.pt`
- held-out Macro L1 为 0.170499，Object 为 0.173084（Object 明显改善约 37.71%），但 Spatial 为 0.167914，出现严重遗忘；所以它不是当前推荐模型。
- 从头训练的 TinyPolicy 仅 213,656 参数，Macro L1 为 0.213461，属于可运行的蒸馏 proof-of-concept，不是最终模型。

## RLinf 多卡速度实验

任务：ManiSkill `PickCube-v1`，state observation，RLinf PPO + MLP policy；单卡 128 个并行环境，四卡 512 个并行环境；每次 rollout 50 steps；均完成 10 个训练更新；关闭 video capture 以绕过服务器无渲染系统问题，单卡/四卡条件一致。

| 配置 | 完整 10 更新耗时 | 稳态单更新 | 完整训练环境步吞吐 | rollout 交互吞吐 |
|---|---:|---:|---:|---:|
| 1 GPU | 70.31 s | 6.415 s | 998 env-step/s | 3,077 env-step/s |
| 4 GPU | 80.44 s | 7.367 s | 3,475 env-step/s | 12,256 env-step/s |
| 4 GPU / 1 GPU | 1.14x wall time | — | **3.48x** | **3.98x** |

四卡每轮处理的环境步数是单卡的四倍，因此虽然单轮墙钟时间略高，单位时间完成的环境交互显著增加。差距主要来自 actor 训练、FSDP 同步和四卡进程调度；rollout 本身接近线性扩展。

训练回报也正常上升：单卡训练回报 2.511 → 7.402，评估回报最高 9.611；四卡训练回报 2.665 → 7.860，评估回报最高 9.748。这里是 PickCube reward/return，不等同于 LIBERO VLA 成功率。

## 原始日志位置

- 单卡完整日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/17_mlp_1gpu/run.log`
- 单卡指标日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/17_mlp_1gpu/metrics.log`
- 四卡完整日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/19_mlp_4gpu/run.log`
- 四卡指标日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/19_mlp_4gpu/metrics.log`
- 四卡第一次 batch 配置校验失败日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/18_mlp_4gpu/run.log`
- VLA RLinf smoke 最后一次日志：
  `/share/yangpengju-local/openvla-oft-repro/runs/exp005/09_rlinf_vla_smoke/run.log`
- 离线后训练总报告：
  `/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260922/REPRODUCTION_REPORT.md`

## VLA 在线 RLinf 的边界和原因

已完成 RLinf VLA worker/model 初始化和多次 smoke；没有把失败结果包装成成功。服务器上的 LIBERO EGL `eglQueryDevicesEXT()` 返回空设备，ManiSkill visual mode 还遇到 Vulkan render system 不可用；最后一次 VLA smoke 还命中了缺失的 Bridge 外部资产 `bridge_v2_real2sim`。这些属于服务器图形驱动/外部资产边界，按服务器规范没有安装驱动、修改系统或越过权限范围，而是转用可复现的离线 VLA 后训练与 state-only RLinf throughput 基准。

## 当前服务器文件框架

根目录：`/share/yangpengju-local/openvla-oft-repro`

```text
cache/                 HuggingFace、pip、xdg 缓存
checkpoints/           官方 OpenVLA-OFT 7B checkpoint
config/libero/         LIBERO 配置
datasets/              libero 与 modified_libero_rlds
logs/                  历史日志
runs/
  post_training_20260922/  审计、数据、SFT、head distill、LoRA smoke、2.04B distill
  post_training_20260923/  replay、Object LoRA rw4/rw8、RLinf VLA smoke
  post_training_20260924/  offline AWR LoRA
  exp005/                  RLinf VLA smoke、兼容层、1/4 GPU speed benchmark
src/LIBERO/
src/openvla-oft/
```

当前没有残留的训练进程，四张 RTX 3090 均已释放；失败的 smoke 目录保留，便于追溯，没有删除用户历史实验。
