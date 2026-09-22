# 新对话长 Prompt：OpenVLA-OFT 离线后训练夜间实验

你是一名严谨的机器人学习复现与后训练工程师。请直接接管服务器，在服务器用户手册允许的范围内，持续推进约 10–12 小时，目标是在已经跑通 OpenVLA-OFT 官方样例推理的基础上，至少完成一种可验证的离线后训练，并尽可能继续完成以下三类方法：

1. 冻结 VLA backbone，只训练 continuous action head 与 proprio projector；
2. 以官方 LIBERO-Spatial checkpoint 为 teacher、用于多场景学习时保留旧能力的 clean knowledge distillation；
3. 受限硬件条件下的 LoRA 后训练。

本轮不再尝试修复 LIBERO/MuJoCo 图形渲染。服务器缺少可用 NVIDIA EGL vendor，且 `/dev/dri/renderD*` 权限不可用；不要把今晚浪费在 EGL、OSMesa、Xvfb、驱动或图形权限上。所有训练和评价都使用官方 RLDS 轨迹中的已有图像、proprio、语言和动作标签，不需要仿真渲染。

请把本 prompt 当成执行契约。你已获得用户授权，可以从 Mac 通过 `ssh act-server` 直接控制服务器、创建小型脚本、启动训练、监控、恢复、停止失败任务、下载服务器端数据以及把必要的轻量结果拉回本地。除非遇到需要用户密码、Hugging Face 登录、Token、管理员权限或会改变实验目标的关键选择，否则不要因普通工程问题停下来等待用户；应在安全边界内自行诊断并继续。

---

## 1. 最终目标与允许的结论

今晚至少要交付一项真正完成的后训练实验。最低完成标准是：

- 有固定的离线 train/held-out 划分；
- 有官方 checkpoint 的未训练 baseline；
- 完成 forward/backward，确认目标参数实际更新、冻结参数未更新；
- checkpoint 能保存并恢复；
- loss 有限并在训练中总体下降；
- 在固定 held-out 数据上重新评价；
- 输出 baseline 与 post-trained 模型的配对指标对照；
- 所有命令、配置、日志、版本、运行时间和偏离官方协议的地方可追溯。

期望形成的严谨表述是：

> 完成了 OpenVLA-OFT 官方样例推理复现；在多个官方 LIBERO RLDS task suite 的 clean 轨迹上进行了离线后训练，并在固定 held-out 离线动作预测指标上获得了改进。由于服务器缺少 EGL 图形运行时，未评价闭环 LIBERO success rate。

绝对不能声称：

- 已完整复现 LIBERO 500-trial 闭环评价；
- 离线 action L1 改善等价于任务成功率改善；
- 受限硬件配置等价于论文的 8×A100、150K-step 官方训练；
- 只凭训练 loss 下降就证明泛化或真实机器人性能提升。

本项目用于快速学习，不要求论文级三 seed、置信区间或严格盲测。允许在固定 held-out split 上比较少量 checkpoint 并选择表现最好的一个，但必须保留所有候选结果，并把结论写成“探索性离线结果”。不能删高 loss 样本、混用不同 split 或把训练 loss 当成 held-out 提升。

---

## 2. 开始前必须完整阅读

首先在 Mac 本地完整阅读：

```text
/Users/yangda/Documents/TaskandWork/embodied_introduction/服务器用户手册.md
/Users/yangda/Documents/TaskandWork/embodied_introduction/openvla/NEW_CHAT_PROMPT.md
```

然后在服务器只读检查并阅读：

```text
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/README.md
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/SETUP.md
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/LIBERO.md
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/vla-scripts/finetune.py
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/prismatic/vla/datasets/
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/experiments/robot/openvla_utils.py
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/experiments/robot/libero/run_libero_eval.py
/share/yangpengju-local/openvla-oft-repro/runs/phase_b_inference/sample_inference_local.log
```

权威资料只优先使用：

- https://github.com/moojink/openvla-oft
- https://openvla-oft.github.io/
- https://arxiv.org/abs/2502.19645
- https://github.com/moojink/openvla-oft/blob/main/SETUP.md
- https://github.com/moojink/openvla-oft/blob/main/LIBERO.md
- https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial
- https://huggingface.co/datasets/openvla/modified_libero_rlds
- https://github.com/Lifelong-Robot-Learning/LIBERO

不要优先根据博客、二手教程或随意的 fork 改变训练语义。

---

## 3. 已确认的复现状态

服务器：

```text
SSH alias: act-server
user: yangpengju
host: eva12-yangpengju / 10.1.12.26
GPU: 4 × NVIDIA GeForce RTX 3090, 24 GB each
driver: 535.179
host CUDA capability: 12.2
```

独立复现环境：

```text
Python: /share/yangpengju-local/anaconda3/envs/openvla-oft-repro/bin/python
Python version: 3.10.14
PyTorch: 2.2.0+cu121
Transformers: 4.40.1, OpenVLA-OFT custom fork
PEFT: 0.11.1
NumPy: 1.26.4
```

代码：

```text
OpenVLA-OFT source:
/share/yangpengju-local/openvla-oft-repro/src/openvla-oft
commit: e4287e94541f459edc4feabc4e181f537cd569a8

LIBERO source:
/share/yangpengju-local/openvla-oft-repro/src/LIBERO
commit: 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

官方 checkpoint：

```text
source: moojink/openvla-7b-oft-finetuned-libero-spatial
known HF revision from the prior download: 6d0231af0e48c5985f1ff86908f4674b84bc049b
server path:
/share/yangpengju-local/openvla-oft-repro/checkpoints/openvla-7b-oft-finetuned-libero-spatial
```

阶段 B 已成功：

```text
VLA: OpenVLAForActionPrediction
processor: PrismaticProcessor
action head: L1RegressionActionHead
proprio projector: ProprioProjector
action chunk shape: (8, 7)
action dtype: float64
single inference time: about 1.066 s
inference peak VRAM: about 14.98 GB
finite: true
range: approximately [-0.000433, 1.0078125]
```

阶段 C 未完成：LIBERO benchmark 和任务定义可以导入，但 MuJoCo offscreen environment 无法创建。不要把这一状态改写为评价成功。

环境曾为兼容 TensorFlow/RLDS 做过以下修复，必须保留并记录：

- NumPy 固定为 1.26.4；
- OpenCV 使用 4.9.0.80 headless runtime；
- TensorFlow 2.15.0、TFDS 4.9.3；
- tensorflow-metadata 1.15.0、protobuf 3.20.3；
- wandb 0.15.12；
- 独立环境内尝试过用户态 Mesa/libglvnd，Conda 可能提示 malformed prefix record。

这些图形包警告不应阻止离线训练，但要重新执行 `pip check`、Python import 和 CUDA matmul smoke，并把结果记录下来。不要为了清除警告重建整个环境，除非环境确实无法训练。

---

## 4. 服务器操作规范：不可越界

必须遵守 `服务器用户手册.md`，特别是：

1. 禁止安装或更新 NVIDIA driver。
2. 禁止 `sudo apt-get upgrade`，禁止修改内核、主机驱动或 NVIDIA runtime。
3. 本轮原则上不使用 sudo；缺包优先装进独立 Conda 环境或使用项目内用户态实现。
4. 不修改其他用户环境，不覆盖 `/share/public-local/opt`。
5. 不把数据集、频繁写入的 checkpoint 或大量小文件放到 NFS。
6. 数据、缓存、checkpoint、日志都放 `/share/yangpengju-local/openvla-oft-repro` 下的明确目录。
7. 不把 token、密码、私钥、W&B 凭据写进脚本、日志、prompt 或 Git。
8. 如果 Hugging Face 需要认证，只提示用户登录；不要索要、读取或输出 token。
9. 不删除旧 checkpoint、旧日志、已有环境或其他实验产物。新实验使用新目录。
10. 不对脏工作树执行 reset、clean、checkout 或盲目 pull。
11. 长任务必须可脱离 Mac 终端运行，并具有 timeout、PID/会话名、日志和退出状态文件。
12. 只在确认 GPU 空闲后使用；不要杀死无法确认属于本实验的进程。

数据应优先由服务器直接下载到个人本地盘。只有在服务器确实无法下载时，才允许 Mac 作为一次性中转；Mac 的下载加上传总流量必须不超过 10 GB。由于中转同一文件通常产生约 `2 × payload_size` 流量，所以中转数据净大小必须控制在 5 GB 以内，并在传输前用精确字节数计算预算。不得用 Mac 中转完整 10.2 GB 数据仓库或 15 GB 模型。Mac 日常只传小型脚本、配置、JSON、CSV、Markdown 和压缩日志。

---

## 5. 目录约定

不要把数据、缓存、日志和 checkpoint 放进 Git 仓库。建议创建：

```text
REPRO_ROOT=/share/yangpengju-local/openvla-oft-repro
SRC=$REPRO_ROOT/src/openvla-oft
PY=/share/yangpengju-local/anaconda3/envs/openvla-oft-repro/bin/python
CHECKPOINT=$REPRO_ROOT/checkpoints/openvla-7b-oft-finetuned-libero-spatial
DATA_ROOT=$REPRO_ROOT/datasets/modified_libero_rlds
HF_HOME=$REPRO_ROOT/cache/huggingface
PIP_CACHE_DIR=$REPRO_ROOT/cache/pip
RUN_ROOT=$REPRO_ROOT/runs/post_training_20260922
CODE_ROOT=$REPRO_ROOT/post_training_20260922
```

每个方法必须有独立目录，例如：

```text
$RUN_ROOT/00_audit
$RUN_ROOT/01_dataset
$RUN_ROOT/02_offline_baseline
$RUN_ROOT/03_head_proprio_sft
$RUN_ROOT/04_head_proprio_distill
$RUN_ROOT/05_lora
$RUN_ROOT/06_final_eval
$RUN_ROOT/manifests
```

脚本可以先在本地 `openvla/` 编写，再通过小型 `scp` 推到 `$CODE_ROOT`；也可以直接在服务器新建 `$CODE_ROOT`。不要修改官方源码，除非确有必要。优先通过独立脚本 import 官方模块。若必须 patch，保存 unified diff、原因和影响范围，不要提交或覆盖原文件。

---

## 6. 今晚的科学问题、场景扩展与指标

主问题：

> 在不能运行闭环仿真的条件下，能否在 clean RLDS 轨迹上继续后训练官方 LIBERO-Spatial checkpoint，使其保持 Spatial 能力的同时，适配更多物体、目标和长时序场景，并降低多个 task suite 的 held-out action prediction L1？

本轮明确不加入像素噪声、动作噪声、亮度扰动或人为 corruption。训练和评价都使用 clean 图像。泛化来自更多任务、更多语言指令和更多物体/目标组合，而不是噪声增强。

官方数据候选：

```text
libero_spatial_no_noops   # 已有 checkpoint 的原始 suite，10 tasks
libero_object_no_noops    # 第一优先扩展，增加物体变化，10 tasks
libero_goal_no_noops      # 第二优先扩展，增加目标变化，10 tasks
libero_10_no_noops        # 第三优先扩展，长时序任务，10 tasks
```

官方代码已定义等权四 suite mixture：

```text
libero_4_task_suites_no_noops =
  spatial + object + goal + libero_10, each sampling weight 1.0
```

下载与训练采用分级策略，不要为了等齐所有数据而耽误最低交付：

1. 先取得 `spatial + object`，形成 20-task clean 多场景实验；
2. 服务器下载顺畅且磁盘足够时，再加入 `goal`；
3. 时间充足时下载全部四个 suite，并直接使用官方 `libero_4_task_suites_no_noops` mixture；
4. 如果只能取得 Spatial，仍完成 Spatial clean 后训练，但明确说明未实现跨 suite 泛化。

Hugging Face 数据仓库总大小约 10.2 GB，Spatial 子目录约 1.91 GB。服务器可直接下载全量；Mac 中转时净 payload 必须不超过 5 GB，优先只中转 Spatial，或在核对精确大小后中转 Spatial + Object。

主要评价：

1. `macro_task_normalized_action_l1`：先计算每个 language task 的 normalized action L1，再对任务等权平均，避免长 episode/task 支配均值；越低越好。
2. `macro_suite_normalized_action_l1`：先对每个 suite 内任务等权平均，再对已下载 suite 等权平均；越低越好。
3. `spatial_retention_l1`：Spatial held-out L1，用于观察增加场景后是否遗忘原 checkpoint 能力。
4. `new_suite_adaptation_l1`：Object/Goal/LIBERO-10 的 held-out L1，用于观察新增场景的适配幅度。

次要评价：

- 每个 suite、每个 task 的 L1；
- 每个 action dimension 的 normalized MAE；
- action chunk 第一步与完整 8 步的 L1；
- 如果 dataset statistics 能可靠恢复，可附加 unnormalized MAE；
- 训练耗时、峰值显存、吞吐量和 trainable parameter 数量。

不要求 bootstrap、论文级置信区间或三 seed。优先用固定 seed `20260922` 完成一个完整闭环；时间宽裕再补第二个 seed。正增长优先定义为宏平均 clean held-out L1 下降，同时单独报告 Spatial 是否发生遗忘。

---

## 7. 数据划分与快速学习模式

优先按 episode 划分，避免相邻帧同时出现在训练和 held-out 集。为了今晚快速完成，不要求复杂的 train/validation/test 三层协议；使用固定 `train/heldout` 即可：

```text
split seed: 20260922
train: 90% episodes per task
heldout: 10% episodes per task
```

如果官方 RLDS pipeline 难以按 episode 注入自定义 split，可以使用每个 task 的官方确定顺序，将最后若干完整 episode 作为 heldout。不能简单随机抽 transition。保存：

```text
$RUN_ROOT/01_dataset/split_manifest.json
```

manifest 至少包含 dataset revision、suite、task/language、episode ID、split 和 transition count。无需为学习实验实现复杂数据库或密码学审计。

建议计算预算：

```text
train transitions cap: 8192–32768 across available suites
heldout transitions: 256–1024 per suite, task-balanced where possible
```

可以用 heldout 指标选择 2–4 个 checkpoint 中的最佳者，但要保存所有候选结果，并明确这是探索性 heldout，不是论文级 untouched test。

---

## 8. 执行顺序与时间门槛

### Stage 0：只读审计，目标 20–30 分钟

执行并记录：

```text
whoami
hostname
date
pwd
umask
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
df -h
df -h /share/yangpengju-local
git SHA/status for OpenVLA-OFT and LIBERO
Python/PyTorch/CUDA/package versions
pip check
checkpoint file listing, sizes, hashes where practical
```

确认：

- GPU 没有其他用户任务；
- `/share/yangpengju-local` 空间足够；
- 当前 checkpoint 包含哪些 action head、proprio projector、dataset statistics、LoRA adapter 或 merged weights 文件；
- 从文件名确定官方 checkpoint step，不能猜测 `150000`；
- 当前 sample inference 仍然可复跑。

保存 `$RUN_ROOT/00_audit/` 下的完整日志、环境快照和 `pip-freeze.txt`。

### Stage 1：服务器直接下载 clean 多场景数据，目标 30–120 分钟

优先使用服务器已有 `huggingface_hub`。第一批先取：

```text
libero_spatial_no_noops/**
libero_object_no_noops/**
README.md
```

这批数据到位后立即开始 dataset smoke 和代码开发；若服务器带宽正常，可在后台继续下载 `libero_goal_no_noops/**` 和 `libero_10_no_noops/**`。推荐用 `snapshot_download(..., repo_type="dataset", allow_patterns=[...])`，固定 `HF_HOME`，记录 resolved revision、实际字节数和文件数。

如果服务器访问 huggingface.co 超时：

1. 先确认不是短暂 DNS/TLS 问题；
2. 可以使用可信 HF 镜像，但必须记录镜像 endpoint，并核对文件结构、大小和 revision/哈希；
3. 若确认服务器无法下载，可以让 Mac 中转，但必须先计算 `2 × payload_bytes <= 10 GB`；优先中转 Spatial，若 Spatial + Object 仍满足预算才可一起中转；
4. Mac 下载目录必须位于本项目 `openvla/transfer/` 或用户明确的临时目录，使用可续传方式上传，传完核对大小/哈希；
5. 如果需要认证，只提示用户执行 HF 登录，绝不要求 token。

下载完成后做 dataset smoke：只加载一个 episode/一个 batch，输出并断言：

```text
full/primary image shape and dtype
wrist image shape and dtype
proprio shape and dtype
actions shape == (8, 7) after batch transform
language instruction present
all numeric tensors finite
normalization statistics present for each loaded suite
dataset/suite identifier retained in the sample
```

### Stage 2：构建离线评价器并冻结 baseline，目标 60–120 分钟

实现独立的 `offline_eval.py`，尽量复用官方：

- processor；
- `RLDSBatchTransform`；
- `PaddedCollatorForActionPrediction`；
- action head；
- proprio projector；
- 官方 action masks 与 parallel action chunk 逻辑。

评价器必须从官方 fine-tuned checkpoint 加载已有 action head 与 proprio projector，不能随机初始化。动作比较应使用与官方训练相同的 normalized action 空间。记录输入图像顺序：third-person 在前，wrist 在后；记录 proprio 维度和 action normalization key。

先在很小的 heldout subset 做代码 smoke，再对每个已下载 suite 运行固定 heldout baseline。保存逐样本结果：

```text
sample_id
episode_id
task/language
suite/dataset name
ground_truth action chunk
predicted action chunk
clean L1
finite flags
inference time
```

建议保存 JSONL 和汇总 JSON。至少输出 per-task、per-suite、macro-task 与 macro-suite L1，不要只保存一个全局均值。

### Stage 3：最高优先级后训练——head + proprio SFT

这是今晚必须优先完成的最低交付。

训练方式：

- 从官方 LIBERO-Spatial fine-tuned checkpoint 初始化；
- 冻结 VLA/vision/language backbone；
- 加载并训练已有 `L1RegressionActionHead` 和 `ProprioProjector`；
- 明确检查 optimizer 参数列表只包含目标模块；
- 使用 ground-truth normalized action chunk 的 L1；
- 训练只使用 clean 图像；多场景来自 Spatial/Object/Goal/LIBERO-10 的真实任务差异；
- `num_images_in_input=2`、`use_proprio=True`、`use_l1_regression=True`、`use_diffusion=False`、`use_film=False`；
- action chunk 保持 8×7，不改变分辨率或动作定义。

官方 `vla-scripts/finetune.py` 强制 `use_lora=True`，因此 head+proprio-only 建议用独立训练脚本复用官方 forward/data components，而不是欺骗官方 CLI。必须正确处理：

- backbone `requires_grad=False`；
- 为了训练 proprio projector，梯度仍需从 VLA 输出传回 projector；
- 如果显存不足，先启用 gradient checkpointing 或减小 micro-batch；
- 若仍不足，允许降级为 action-head-only，并明确标注，不得声称 projector 已训练；
- 不要随机重置官方 head/projector。

建议起始配置，允许 agent 根据 smoke 的显存和 loss 做一次受控调整：

```text
micro batch: 1
gradient accumulation: 8
effective batch: 8
optimizer: AdamW
learning rate: 1e-4 for head/projector; if unstable use 5e-5
weight decay: 0
warmup: 50 steps or 5%
smoke: 5 steps
short run: 250 steps
formal overnight run: 1000–3000 optimizer steps depending measured throughput
validation frequency: 100–250 steps
checkpoint frequency: 250–500 steps
precision: bf16 where supported
seed: 20260922 initially
```

smoke 必须验证：

- 一个 batch forward/backward；
- loss/gradient finite；
- head 参数确实变化；
- proprio projector 参数确实变化；
- backbone 参数 checksum/selected tensors 不变；
- checkpoint save；
- 新进程 resume 后再训练 1–2 步；
- 峰值显存低于硬件上限并保留安全余量。

formal run 只能在 smoke 全部通过后开始。优先按 clean heldout `macro_suite_normalized_action_l1` 选择 checkpoint，同时检查 Spatial retention。先完成 seed `20260922`；有余量再做第二个 seed，不强求三 seed。

### Stage 4：高优先级——teacher consistency distillation

用户对蒸馏感兴趣。这里采用 clean multi-suite continual adaptation / learning-without-forgetting 形式：

- teacher：固定的官方 fine-tuned LIBERO-Spatial checkpoint；
- teacher 只为 Spatial clean 样本生成 action chunk；
- student：从同一官方 checkpoint 初始化，在 Spatial + 新 suite 上继续训练；
- ground-truth action 是所有 suite 的主要监督；
- teacher target 只作为 Spatial 样本上的保持项，减少新增场景训练导致的 Spatial 遗忘；
- teacher 永远 `eval()`、`no_grad()`，不可更新；
- Object/Goal/LIBERO-10 不强行使用 Spatial teacher 软标签，因为 teacher 未针对这些 suite 训练。

推荐预注册损失：

```text
L_gt = L1(student(clean_obs), ground_truth_action)
L_keep = SmoothL1(student(clean_spatial_obs), stopgrad(teacher(clean_spatial_obs)))

L_total = L_gt + 0.25 * I[suite == spatial] * L_keep
```

优先缓存 teacher action chunks 到服务器本地，避免训练时同时驻留两个 7B 模型。cache 至少带 checkpoint revision、sample ID、normalization space、dtype 和 shape。无需做复杂密码学审计，但必须能检测错位或缺失 sample。

为了判断蒸馏本身是否贡献改进，若时间允许，比较：

```text
A: clean multi-suite head+proprio SFT, no teacher loss
B: same setup + Spatial teacher retention loss
```

除 teacher loss 外保持数据、seed、步数、优化器一致。若时间不足，先完成 B，但必须说明没有完成消融。

### Stage 5：LoRA 后训练

只有在至少一项 head/proprio 训练完成并有 checkpoint 后，才允许投入大量时间调 LoRA。

官方配置说明：batch size 1 约需 25 GB/卡，而 RTX 3090 只有 24 GB；4 卡 DDP 不会降低单卡模型内存。因此不得直接声称官方 rank-32 配置可运行。

LoRA 从已经 fine-tuned 的官方 checkpoint 开始继续适配，不从随机 action head 开始。重点审计官方 `finetune.py` 的 resume 逻辑：如果直接把 `vla_path` 指向 merged checkpoint，但未正确加载已有 action head/proprio projector，脚本会重新初始化这些模块，这是不可接受的。必须通过实际 checkpoint 文件名确定 `resume_step`，或实现显式 loader。

推荐尝试顺序：

1. LoRA rank 8，micro-batch 1，bf16，gradient accumulation 8；
2. `merge_lora_during_training=False`，避免保存时额外加载完整 base model；
3. 启用 model 支持的 gradient checkpointing；
4. 如果仍 OOM，评估冻结 vision backbone、8-bit optimizer 或 QLoRA 4-bit；
5. 所有量化、rank 降低、gradient checkpointing、冻结范围变化都记录为 constrained adaptation。

先单卡 3–5 step smoke。DDP 多卡只用于吞吐，不应被当作显存解决方案。若 60–90 分钟内仍无法在 24 GB 上稳定完成 forward/backward/save/resume，则停止 LoRA 调试，保存错误日志，转回已可行的方法；不要牺牲最低交付。

若 LoRA smoke 成功：

- 确认 adapter 参数确实更新；
- head/proprio 是否同时更新必须明确；
- 运行 500–2000 steps，具体取决于吞吐与剩余时间；
- adapter 单独保存；
- 合并不是完成训练的必要条件，若合并耗时/内存高可延后；
- offline evaluator 必须能加载 adapter + head + proprio，不能只评价未加载 adapter 的 base model。

---

## 9. Luna 必须遵循的代码框架与技术细节

Luna 不要把全部逻辑塞进一个临时脚本。建议在本地 `openvla/nightly/` 和服务器 `$CODE_ROOT` 保持同样结构：

```text
nightly/
  config.py
  checkpoint_io.py
  data.py
  forward.py
  offline_eval.py
  train_head_proprio.py
  build_teacher_cache.py
  train_distill.py
  train_lora.py
  launch.py
  summarize.py
  tests/
    test_checkpoint_load.py
    test_batch_shapes.py
    test_forward_equivalence.py
    test_save_resume.py
```

不要复制整个官方仓库。独立脚本通过 `PYTHONPATH=$SRC` import 官方模块。

### 9.1 `config.py`

使用 dataclass 或 JSON 配置，不要依赖散落的 shell 常量。至少包含：

```python
@dataclass
class ExperimentConfig:
    checkpoint_path: str
    data_root: str
    run_dir: str
    suites: tuple[str, ...]
    suite_weights: dict[str, float]
    seed: int = 20260922
    num_images_in_input: int = 2
    use_proprio: bool = True
    action_chunk: int = 8
    action_dim: int = 7
    proprio_dim: int = 8
    micro_batch_size: int = 1
    grad_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 1500
    eval_freq: int = 250
    save_freq: int = 250
    distill_weight: float = 0.25
```

启动时把完整 resolved config 写入 run 目录。任何 CLI override 也写入日志。

### 9.2 `checkpoint_io.py`

加载顺序必须固定：

1. 注册本地 `OpenVLAConfig`、processor 和 model AutoClass；
2. 从本地 merged checkpoint 加载 processor 和 VLA；
3. 设置 `vla.vision_backbone.set_num_images_in_input(2)`；
4. 使用官方 `get_action_head` / `get_proprio_projector` 或按实际文件名显式加载现有权重；
5. 对 action head/proprio 的 `missing_keys` 和 `unexpected_keys` 做严格检查；
6. 打印每个模块参数量、dtype、device 和 trainable count；
7. 保存 checkpoint provenance，包括官方模型 revision 和组件文件名。

最危险的错误是 VLA 加载正确，但 action head 或 proprio projector 被随机初始化。必须设置一个 probe batch，在任何训练前比较：

- 独立 loader 的预测；
- 已跑通 sample inference 路径的预测。

两条路径的 action shape 必须都是 `(8, 7)`，数值应在浮点误差内一致。若不一致，先修 loader，禁止训练。

自定义训练 checkpoint 建议格式：

```python
{
    "step": step,
    "action_head": action_head.state_dict(),
    "proprio_projector": proprio_projector.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "config": asdict(cfg),
    "dataset_statistics": stats,
    "suite_names": suite_names,
    "official_checkpoint_revision": revision,
}
```

LoRA adapter 单独使用 `save_pretrained()` 保存，不要把 15 GB merged model 每 250 steps 重写一遍。

### 9.3 `data.py`

优先复用：

```python
RLDSBatchTransform
RLDSDataset
PaddedCollatorForActionPrediction
PurePromptBuilder
ActionTokenizer
```

官方四 suite mixture 已在 `OXE_NAMED_MIXTURES` 注册。如果四个目录都存在，直接使用：

```python
dataset_name = "libero_4_task_suites_no_noops"
```

如果只有 Spatial + Object，不要伪造一个数据目录。可以在独立脚本中构造局部 mixture specification：

```python
mixture = [
    ("libero_spatial_no_noops", 1.0),
    ("libero_object_no_noops", 1.0),
]
```

然后调用官方 interleaved dataset builder；如果 `RLDSDataset` 只接受注册名，可在进程内向 `OXE_NAMED_MIXTURES` 添加一个局部 key，例如 `libero_spatial_object_no_noops_local`，但不要修改官方源码文件。所有 suite 等权采样，避免大数据集吞没小数据集。

batch 必须保留 `dataset_name`、task/language 或可推导 sample ID。确认 official transform 使用：

```text
primary image key: image
wrist image key: wrist_image
proprio source: EEF_state + gripper_state
normalization: bounds_q99
```

记录真实 batch shape，不要凭猜测硬编码 pixel channel layout。必须断言：

```python
actions.shape[-2:] == (8, 7)
proprio.shape[-1] == 8
torch.isfinite(actions).all()
torch.isfinite(proprio).all()
```

TensorFlow 和 PyTorch 同进程时，防止 TensorFlow 预占 GPU：在 import TensorFlow 后枚举 GPU 并启用 memory growth，或在数据进程中对 TensorFlow 隐藏 GPU。不要因此让 PyTorch 看不到指定 GPU。

`shuffle_buffer_size=100000` 可能消耗较多 RAM；先用 1000 做 smoke，正式训练视内存用 10000–50000。RLDS DataLoader 维持 `num_workers=0`，因为官方 loader 自己管理并行。

### 9.4 `forward.py`

必须最大程度复制官方 `run_forward_pass()` 的语义。核心张量流为：

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = vla(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
        labels=batch["labels"],
        output_hidden_states=True,
        proprio=batch["proprio"] if use_proprio else None,
        proprio_projector=proprio_projector if use_proprio else None,
        use_film=False,
    )

ground_truth_token_ids = batch["labels"][:, 1:].to(device)
current_mask = get_current_action_mask(ground_truth_token_ids)
next_mask = get_next_actions_mask(ground_truth_token_ids)
text_hidden = output.hidden_states[-1][:, num_patches:-1]
action_hidden = text_hidden[current_mask | next_mask]
action_hidden = action_hidden.reshape(batch_size, 8 * 7, vla.llm_dim)
pred_actions = action_head.predict_action(action_hidden.to(torch.bfloat16))
pred_actions = pred_actions.reshape(batch_size, 8, 7)
loss = torch.nn.functional.l1_loss(pred_actions, gt_actions)
```

实际 `L1RegressionActionHead.predict_action()` 是否已返回 `(B,8,7)` 要通过 shape 日志确认，不要无条件重复 reshape。`num_patches` 必须是：

```text
vision patches per image × 2 images
+ 1 proprio embedding
```

错误的 `num_patches` 会导致 action token hidden state 错位，即使 loss 仍能计算，也会使实验无效。

### 9.5 `train_head_proprio.py`

冻结方式：

```python
vla.requires_grad_(False)
action_head.requires_grad_(True)
proprio_projector.requires_grad_(True)
```

注意：训练 proprio projector 时不能把整个 VLA forward 包在 `torch.no_grad()` 中，因为 projector 输出经过 VLA 后需要反向传播。VLA 参数可以冻结，但 autograd graph 仍需保留。如果这导致 OOM：

1. micro-batch 保持 1；
2. 开启 VLA gradient checkpointing；
3. 降低 sequence 中非必要输出；
4. 最后才降级为 action-head-only。

只有 action-head-only 时，才允许在 `torch.no_grad()` 下缓存 `action_hidden`，之后用轻量 PyTorch DataLoader 快速训练 head。这个 fallback 很适合保证今晚至少完成一个后训练闭环。

optimizer 只接收明确列出的参数：

```python
trainable = list(action_head.parameters()) + list(proprio_projector.parameters())
optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=0.0)
```

训练前后分别计算选定参数 tensor 的 SHA256/最大绝对差：

- head 至少一个 tensor `max_abs_delta > 0`；
- projector 至少一个 tensor `max_abs_delta > 0`；
- backbone probe tensors `max_abs_delta == 0`。

每次 accumulation 前将 loss 除以 `grad_accumulation_steps`。在 optimizer step 前记录 global grad norm，并在需要时 `clip_grad_norm_(trainable, 1.0)`。日志至少记录 step、LR、train L1、heldout L1、grad norm、examples/s、GPU allocated/reserved memory。

### 9.6 `build_teacher_cache.py` 与 `train_distill.py`

teacher cache 只覆盖 Spatial train samples。teacher 必须使用官方 action head 和 proprio projector，输出 normalized action chunk：

```text
sample_id -> float16[8, 7]
```

用 `.npz`、safetensors 或按 shard 保存，避免数万个小文件。每个 shard 附 sample ID 列表。训练时检查 sample ID 完全匹配；找不到 teacher target 的非 Spatial 样本只计算 ground-truth loss。

distillation loss：

```python
gt_loss = F.l1_loss(student_actions, gt_actions)
if suite == "libero_spatial_no_noops":
    keep_loss = F.smooth_l1_loss(student_actions, teacher_actions)
else:
    keep_loss = 0.0
loss = gt_loss + 0.25 * keep_loss
```

mixed batch 内按样本 mask 计算，而不是用 Python 的单一 `if` 假设整个 batch 同 suite。由于 micro-batch 可能为 1，上述逻辑仍需支持将来扩大 batch。

### 9.7 `train_lora.py`

LoRA 必须基于官方已 fine-tuned merged VLA，而不是重新下载并从 `openvla/openvla-7b` 开始。建议：

```python
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules="all-linear",
    init_lora_weights="gaussian",
)
student_vla = get_peft_model(official_merged_vla, lora_cfg)
```

显式加载官方 action head/proprio projector。不要直接使用会随机创建新 head 的代码路径。若同时训练 LoRA + head + projector，optimizer 参数组可以使用不同 LR：

```text
LoRA: 5e-5
action head + proprio projector: 1e-4
```

如果 rank 8 仍 OOM，可尝试只对 language model linear layers 加 LoRA，而不是 vision backbone；再不行才尝试 4-bit QLoRA。不要为了 LoRA 成功而改变 action chunk、图像数量或 proprio 输入。

### 9.8 `offline_eval.py`

评价器使用 `model.eval()`、`torch.inference_mode()`，输入全部是 clean 数据。输出：

```text
per-sample JSONL
per-task metrics
per-suite metrics
macro-task mean
macro-suite mean
first-action-step L1
full-chunk L1
per-dimension MAE
```

多 suite 的 ground-truth actions 各自使用其 dataset statistics 归一化。不要拿 Spatial 的 unnormalization key 去反归一化 Object/Goal/Long。主要比较在 normalized space 完成；只有确定每个 suite stats 对应正确时才附加 unnormalized 指标。

checkpoint reload 验证：同一个 probe batch 在保存前与新进程恢复后的预测 `max_abs_diff` 应足够小（建议 `<1e-4`，若 bf16 导致更大则记录实际值和原因）。

### 9.9 `launch.py` 与 `summarize.py`

launcher 只负责生成 resolved config、command、tmux/nohup 任务和 manifest，不在 import 时启动任务。summarizer 从 JSONL 计算指标，不重新加载 7B 模型。任何失败 run 都保留状态和最后 traceback。

---

## 10. 多 GPU 调度建议

所有正式调度前先确认 GPU 空闲。建议按任务并行而不是盲目 DDP：

```text
GPU 0: per-suite baseline evaluation / Spatial teacher cache
GPU 1: clean multi-suite head+proprio SFT
GPU 2: clean multi-suite distillation run
GPU 3: LoRA memory smoke / checkpoint evaluation
```

实际分配根据显存监控调整。每个进程显式设置 `CUDA_VISIBLE_DEVICES`，日志中同时记录物理 GPU index 和进程内 logical index。不要让 TensorFlow 抢占全部 GPU；如 RLDS/TensorFlow 只用于数据管线，配置合理的内存行为或隐藏不需要的设备，同时确保 PyTorch GPU 可见性正确。

如果 CPU/RAM/磁盘读取成为瓶颈，不要同时启动四个 RLDS 大 shuffle job。先降低 shuffle buffer、错峰启动或共享只读 teacher cache。官方默认 `shuffle_buffer_size=100000` 可因内存限制降低，但必须记录偏离。

---

## 11. 后台运行、监控和恢复

用户可能离开十多个小时。所有长任务必须在服务器端独立存活，优先使用 `tmux`；若无 tmux，使用 `nohup timeout 11h bash ...`。不要依赖 Mac 终端持续连接。

每个任务至少生成：

```text
command.sh
config.json or config.yaml
stdout.log
stderr.log, or merged run.log
pid.txt or tmux session name
STARTED marker
SUCCESS or FAILED marker
exit_code.txt
metrics.jsonl
gpu_memory.csv
```

建议每 10–20 分钟检查一次，而不是高频轮询：

- 进程是否存活；
- 日志最近更新时间；
- loss 是否 finite；
- GPU utilization/VRAM/temperature；
- 磁盘剩余空间；
- checkpoint 是否按预期生成。

遇到 OOM：记录完整 traceback、配置和峰值显存，再按预注册顺序降低资源。遇到 NaN：停止该 run，保留 checkpoint 和日志，检查输入 finite、LR、bf16、gradient norm；不要继续污染结果。遇到 SSH 断开：后台任务不应受影响，重连后继续监控。

可以启动 TensorBoard，但只绑定服务器 `127.0.0.1`；若需要访问，使用 SSH local forwarding。不要把端口暴露到公共网卡。W&B 默认 `WANDB_MODE=offline` 或完全禁用，除非用户已经配置账号；不要要求或输出凭据。

---

## 12. 快速学习模式下的可信比较底线

本项目是个人上手实验，不要求论文级三 seed、bootstrap、盲测或大规模消融，但仍需保证“增长”来自同口径比较，而不是统计口径变化：

- baseline、SFT、distillation、LoRA 使用同一份固定 held-out manifest；
- 各方法使用相同图像预处理、action normalization、suite 权重和样本上限；
- 训练前先保存 baseline 的逐样本预测与汇总；
- 不删高 loss 或失败样本；
- 不把 teacher agreement 或 train loss 当作主要效果指标；
- 不在看到结果后更换 primary metric；
- 如果 action-head-only 改善而 LoRA 下降，两个结果都保留；
- 多场景总体指标改善但 Spatial 退化时，明确报告适配/遗忘权衡；
- 若只跑一个 seed，直接报告该 seed，不制造 mean/std。

为了今晚更容易完成，可以在固定 held-out 上比较至多两档学习率和 2–4 个预先设定的 checkpoint step，并选择其中最好的一个。保存所有候选结果，明确它是“探索性 held-out 选择”，不额外花时间搭建 untouched test。

---

## 13. 每种方法的完成判据

### Head + proprio SFT 完成

- baseline 已测；
- 数据 split 固定；
- 两个模块从官方 checkpoint 正确加载；
- 两个模块参数都更新；
- backbone 未更新；
- checkpoint save/resume 通过；
- fixed held-out evaluation JSON 完整，至少含 per-task、per-suite 和 macro 指标。

如果因显存降级为 head-only，完成状态必须写作 `action-head-only completed; proprio training not completed`。

### Distillation 完成

- teacher 固定且来源明确；
- teacher targets 与 sample manifest 一一对应；
- student loss 实际包含 teacher term；
- 有 ground-truth held-out 指标，而不只是 teacher agreement；
- checkpoint save/resume 和 fixed held-out evaluation 完成。

### LoRA 完成

- adapter 从官方 fine-tuned checkpoint 上新增；
- 原有 head/proprio 被正确加载；
- adapter 参数更新；
- 可重新加载并得到同一离线预测；
- fixed held-out evaluation 完成；
- 所有显存受限改动列为偏离官方协议。

---

## 14. 结果汇总与本地拉取

服务器最终至少生成：

```text
$RUN_ROOT/summary.json
$RUN_ROOT/RESULTS_TABLE.csv
$RUN_ROOT/REPRODUCTION_REPORT.md
$RUN_ROOT/environment.txt
$RUN_ROOT/pip-freeze.txt
$RUN_ROOT/commands.log
$RUN_ROOT/manifests/
```

`summary.json` 至少包含：

- baseline 的 macro-task、macro-suite、per-suite 和 per-task clean held-out L1；
- 每个完成方法的同口径 clean held-out L1；
- 相对 baseline 的 absolute delta 和 relative improvement；
- Spatial retention delta 与每个新增 suite 的 adaptation delta；
- full-chunk、first-step 和 per-dimension MAE；
- 实际 seed；若完成两个 seed 才计算 mean/std；
- train/heldout 的 episode/transition 数与 manifest 标识；
- trainable parameter count；
- training steps、wall time、峰值显存；
- checkpoint 路径；
- 完成/失败状态和失败原因。

实验完成后，把必要轻量结果拉到：

```text
/Users/yangda/Documents/TaskandWork/embodied_introduction/openvla/results/server_20260922/
```

允许拉取：

- Markdown 报告；
- JSON/JSONL 汇总和必要的逐样本指标；
- CSV；
- 配置文件；
- 自编脚本；
- launcher manifest；
- 压缩后的关键日志；
- 小型曲线 PNG。

禁止拉取：

- 15 GB 模型；
- LoRA 合并后的完整 checkpoint；
- RLDS 数据集；
- Hugging Face cache；
- 大型 TensorBoard event；
- 重复图像或视频。

如果 LoRA adapter 或 action head checkpoint 很小且对复现确有必要，先检查大小；默认仍留在服务器，只在 manifest 中记录绝对路径和 SHA256，不自动拉回 Mac。

本地最终补充：

```text
openvla/REPRODUCTION_REPORT.md
openvla/EXPERIMENT_PLAN.md
openvla/results/server_20260922/
```

不要自动 git commit 或 push。检查 `.gitignore`，避免 checkpoint、dataset、cache、event 文件进入 Git。

---

## 15. 推荐的今晚决策树

严格按以下优先级推进：

```text
审计通过
  → 服务器优先下载 Spatial + Object；其余 suite 可后台继续
  → dataset smoke
  → 固定每 task 的 episode-level train/heldout split
  → baseline offline evaluation
  → head+proprio 5-step smoke
  → head+proprio short run + save/resume
  → head+proprio formal run + held-out evaluation
  → 至此达到最低交付
  → 为 Spatial train samples 构建 teacher cache
  → clean retention distillation smoke
  → distillation formal run + held-out evaluation，比较 Spatial 遗忘与新 suite 适配
  → LoRA 3–5 step memory smoke
  → 若可行则 LoRA formal run；若 60–90 分钟仍 OOM 则停止
  → 汇总、拉取轻量结果、完成报告
```

若时间明显不足，宁可完整闭环一项 head+proprio 实验，也不要留下三个只有 forward 的半成品。若 head+proprio 同训 OOM，按顺序退化：

```text
gradient checkpointing
→ 更小 shuffle buffer / micro-batch 1
→ action-head-only frozen-feature adaptation
→ 完成 action-head-only 的完整 baseline/train/heldout 闭环
```

若数据下载阻塞，不要回到图形渲染问题，也不要用单个 sample observation 伪造正式训练。保存网络诊断并报告数据阻塞；只允许做代码/loader smoke，不能称为完成后训练。

---

## 16. 最终向用户汇报的格式

最终回答必须先给结论，再给证据：

1. 今晚完成了哪些方法，哪些未完成；
2. baseline 与 post-training 的 clean held-out 指标表，包含各 suite、macro-task、macro-suite；
3. Spatial retention 和新增 suite adaptation 的变化，以及最佳 checkpoint 的探索性选择规则；
4. 训练步数、耗时、峰值显存和 GPU 使用；
5. checkpoint 在服务器的路径和哈希；
6. 本地拉回了哪些文件；
7. 所有偏离官方协议；
8. 图形渲染仍不可用，因此哪些结论不能下；
9. 失败尝试和下一步最值得做的工作。

推荐最终措辞示例：

> 官方 OpenVLA-OFT LIBERO-Spatial checkpoint 的 sample-observation 推理已复现。随后在可获得的官方 LIBERO RLDS clean task suites 上完成了受限硬件条件下的离线后训练。与冻结的官方 checkpoint 相比，所选模型在固定 held-out 数据上的 macro-suite normalized action L1 从 A 降到 B（相对改善 C%）；Spatial retention L1 从 D 变为 E，新增 suite 的宏平均 L1 从 F 变为 G。该结果证明的是探索性离线动作预测指标改善，不是闭环 LIBERO success rate 改善。

不要使用“全面超过论文”“LIBERO 成功率提高”或“完整复现”等超出证据的表述。

现在开始执行。第一条 commentary 应简要说明你已读完服务器手册、本 prompt 和官方训练入口；然后进行 Stage 0 只读审计。只要操作仍在上述授权和服务器规范内，就持续推进，不要在每个普通步骤重新向用户索取许可。
