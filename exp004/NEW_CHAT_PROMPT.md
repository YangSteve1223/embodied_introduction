# EXP-004 新对话长 Prompt：温和 combined-noise 剂量—反应 PPO continuation

你正在继续指导我的具身智能实验项目。请先完整理解项目、历史实验和本 prompt 中的 EXP-004 预注册设计，再逐步带我操作。

## 1. 协作方式与操作边界

我是初学者。请解释每一步的目的、输入、输出、成功判据和风险。默认由我在 Mac 或服务器终端中执行命令，你提供命令并等待我贴回输出。

在我明确审核同意前，不要：

- 自动连接服务器；
- 自动启动、停止或恢复训练；
- 自动启动正式评价；
- 下载数据集或模型；
- 安装或升级 Python、CUDA、驱动、Vulkan、系统包；
- 修改服务器公共目录或其他用户文件；
- 删除、覆盖或重命名旧实验结果；
- 对服务器脏工作树执行盲目的 `git pull`、reset、clean 或 checkout；
- 把私有结果、服务器信息、checkpoint 或 memory 文件提交到 GitHub。

大型代码文件和新工程文件优先在本地工作区编写、审阅和测试。我会通过 GitHub Desktop 提交/推送，再由我在服务器执行经过审核的同步命令。必要时也可以给出明确、窄范围的 `scp` 命令。不要凭猜测大改旧 PPO 脚本。

本轮新对话的第一阶段只做：阅读、审计、完善规划、设计本地代码框架和测试。完成后停下来等我审核，不启动服务器实验。

## 2. 项目目标

总目标是学习具身智能 post-training，路线为：

```text
可靠的 state-based RL baseline
→ robustness/post-training 方法验证
→ BC / ACT baseline
→ BC→RL post-training
→ KL、replay、蒸馏、offline RL 和其他后训练实验
```

当前仍是 state-only ManiSkill PPO 阶段，不能包装成视觉 VLA 结论。

## 3. 本地与服务器环境

本地项目：

```text
/Users/yangda/Documents/Codex/2026-09-18/wo/embodied_introduction
```

服务器项目：

```text
/share/yangpengju-local/embodied/embodied_introduction
```

服务器实验根目录：

```text
/share/yangpengju-local/embodied/runs
```

服务器环境：

- SSH alias：`ssh act-server`，仅用户 Mac 终端可用；
- Python：`/share/yangpengju-local/anaconda3/envs/maniskill/bin/python`；
- GPU：4 × RTX 3090；
- 只能使用 state-only、`physx_cuda`、`render_backend=none`；
- 不使用 sudo，不修改驱动、内核、Vulkan 或公共环境；
- 训练、日志、checkpoint 只写入 `/share/yangpengju-local`；
- 每个长任务必须有 timeout、独立日志、manifest、明确退出码；
- `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`；
- 后台任务必须显式使用上述 maniskill Python，不能依赖 base Conda；
- 服务器最后一次已知 Git 状态较旧且有 EXP-002 的 modified/untracked 文件，因此必须先重新只读查询，不能直接 pull 或清理。

本地 Git 在生成本 prompt 前的状态：

```text
branch: main
HEAD: 8b13471
working tree: clean（空的 exp004 目录不被 Git 跟踪）
```

## 4. 新对话开始时必须阅读

请按顺序完整阅读：

1. `README.md`
2. `MEMORY.md`
3. `EXPERIMENT_CONTRACT.md`
4. `EXPERIMENT_RECORD.md`
5. `exp003/EXPERIMENT_PLAN.md`
6. `exp003/configs/experiment.json`
7. `exp003/config.py`
8. `exp003/noise.py`
9. `exp003/curriculum.py`
10. `exp003/checkpoint.py`
11. `exp003/continue_ppo.py`
12. `exp003/evaluate.py`
13. `exp003/launch_calibration.py`
14. `exp003/launch_formal_training.py`
15. `exp003/launch_formal_evaluation.py`
16. `exp003/summarize_calibration.py`
17. `exp003/summarize_formal.py`
18. `exp003/smoke.py`
19. `exp003/tests/`
20. `exp003/results/server_20260920/exp003_summary.json`
21. `exp003/results/server_20260920/calibration_summary.json`
22. `exp003/results/server_20260920/calibration_selection.json`

还要检查 `.gitignore`，确保未来的 `exp004/results/`、checkpoint、TensorBoard event 和私有服务器结果不会进入 Git。

## 5. 已完成实验背景

### EXP-001：PickCube state PPO pipeline baseline

EXP-001 证明了 state-based GPU PPO pipeline 可以训练、保存和评价。任务接近饱和，不能据此宣称复杂接触任务、视觉泛化或 post-training 能力。

### EXP-002：PegInsertionSide noisy PPO from scratch

固定接口：

```text
task: PegInsertionSide-v1
robot_uids: panda_wristcam
obs_mode: state
observation dimension: 43
control_mode: pd_ee_delta_pose
action dimension: 7
reward_mode: normalized_dense
sim_backend: physx_cuda
render_backend: none
training seeds: 1001, 1002, 1003
num_envs: 2048
num_steps: 100
total timesteps/run: 102,400,000
target_kl: null
formal evaluation: 256 episodes/cell
```

EXP-002 比较从随机初始化开始的 clean PPO 和 interface-noise PPO。三 seed 汇总：

```text
clean train → clean eval: 41.54% ± 16.11%
clean train → noisy eval: 41.93% ± 17.80%
noisy train → clean eval: 10.29% ± 12.62%
noisy train → noisy eval: 10.94% ± 13.23%
robustness_gain: -30.99 pp
clean_cost: -31.25 pp
```

结论：从头进行当前强度的 noisy PPO 明显失败。EXP-002 是完整负结果，不得删除、改写或包装成成功结果。

### EXP-003：calibrated interface-noise PPO continuation

EXP-003 从同一个 EXP-002 clean checkpoint 和完整 optimizer state 分叉：

```text
control: clean continuation
treatment: combined-noise curriculum continuation
```

校准阶段完成 60/60 个格子：

```text
combined 1x: clean-policy success drop 1.04 pp，不满足门槛
combined 2x: drop 6.84 pp，不满足门槛
combined 4x: drop 20.57 pp，满足门槛并被选中
combined 4x noise-induced clip fraction: 约 0.60%
```

正式 continuation：

```text
additional timesteps: 20,480,000
updates: 100
warmup: updates 1–20
checkpoint interval: 10 updates
seeds: 1001, 1002, 1003
6/6 runs 完成
final global_step: 122,880,000
36/36 formal evaluation cells 完成
```

EXP-003 正式结果：

```text
aggregate robustness_gain: -8.79 pp ± 5.67 pp
aggregate clean_cost: -27.15 pp ± 3.73 pp
positive robustness seeds: 0/3
all_primary_gates_pass: false
```

逐 seed success_once：

```text
seed   original clean/noisy   control clean/noisy   treatment clean/noisy   robustness gain   clean cost
1001   30.66% / 8.59%         50.39% / 33.98%       27.15% / 19.34%         -14.65 pp         -23.24 pp
1002   51.56% / 23.05%        52.54% / 35.74%       21.88% / 27.34%         -8.40 pp          -30.66 pp
1003   52.34% / 41.21%        50.39% / 39.45%       22.85% / 36.13%         -3.32 pp          -27.54 pp
```

结论：4x combined-noise continuation 在三个 seed 上都没有提升 noisy success，并造成显著 clean forgetting。action clipping 很低，因此不能简单把失败归因于动作裁剪。仍未回答的问题是：4x 是否对训练过强，以及 1x–3x 是否存在鲁棒性与 clean retention 的甜点区。

本地 EXP-003 轻量结果归档：

```text
exp003/results/server_20260920/
```

该目录约 4.7 MB、229 个文件，含 60 个 calibration JSON/log、36 个 formal evaluation JSON/log、6 个训练日志/config、manifests 和 summaries；checkpoint 与 TensorBoard events 未拉回本地。服务器原始结果仍位于：

```text
/share/yangpengju-local/embodied/runs/exp003
```

## 6. 为什么今晚不直接转 ACT/BC

ACT/BC 是后续主路线，但当前还没有完成 demonstration 数据、转换、训练入口和评价链路的端到端 smoke。把第一个 BC 工程 smoke 和正式 8–10 小时实验合在同一夜，失败风险过高，也难以区分环境问题、数据问题和算法问题。

EXP-004 应先关闭现有 PPO noise 路线中最关键的因果问题。如果 0x–3x 的温和剂量仍全部失败，则停止继续调 multiplier，下一阶段正式转入 ACT/BC pipeline。

## 7. EXP-004 预注册研究问题

标题：

```text
EXP-004 — Mild combined-noise dose-response PPO continuation
```

研究问题：

```text
EXP-003 的失败是否主要由 combined-noise 4x 强度过大造成？
在相同 source checkpoint、optimizer、PPO 预算和评价协议下，
combined 1x、2x 或 3x continuation 是否能提高共同 4x OOD 条件下的成功率，
同时把 clean success 损失控制在 5 个百分点以内？
```

主要假设：至少一个温和 multiplier 在共同的 combined-4x evaluation 下，相对 clean continuation 获得不低于 +5 pp 的 noisy success 增益，至少 2/3 seeds 同方向，且 clean cost 不差于 -5 pp。

反证条件：如果 1x、2x、3x 在 update 100 和 update 150 均未通过门槛，则当前 interface-noise PPO continuation 路线停止，不再进行 multiplier 微调。

## 8. 实验臂与唯一改变变量

所有新 run 都从对应的 EXP-002 clean final checkpoint 开始，并恢复相同 actor、critic、optimizer、global step 和训练 seed。

```text
Arm C0: combined 0x，clean continuation
Arm C1: combined 1x curriculum continuation
Arm C2: combined 2x curriculum continuation
Arm C3: combined 3x curriculum continuation
Reference C4: 复用 EXP-003 combined 4x update-100 final checkpoints
```

新训练矩阵：

```text
4 arms × 3 training seeds = 12 new runs
training seeds: 1001, 1002, 1003
```

唯一组间变量是 continuation 期间 combined interface-noise 的 multiplier。不要同时改变 noise component、PPO 超参数、网络、reward、task、action controller、evaluation protocol 或 source checkpoint。

C4 只用于 update-100 剂量参考，不重新训练，也不把它伪装成 update-150 结果。

## 9. 固定训练参数

```text
task: PegInsertionSide-v1
robot_uids: panda_wristcam
obs_mode: state
control_mode: pd_ee_delta_pose
reward_mode: normalized_dense
sim_backend: physx_cuda
render_backend: none

num_envs: 2048
num_steps: 100
batch_size: 204,800
learning_rate: 0.0003
entropy_coefficient: 0.01
num_minibatches: 32
update_epochs: 8
gamma: 0.8
gae_lambda: 0.9
clip_coef: 0.2
vf_coef: 0.5
max_grad_norm: 0.5
target_kl: null

continuation_updates: 150
additional_timesteps: 30,720,000
source_global_step: 102,400,000
update-100 global_step: 122,880,000
update-150 global_step: 133,120,000
checkpoint_interval_updates: 10
```

噪声 curriculum：

- C0 始终为 0x；
- C1/C2/C3 在 updates 1–20 从 0 线性增至目标 multiplier；
- updates 21–150 保持目标 multiplier；
- warmup 固定为 20 updates，不能因为总训练长度变为 150 而改成 30 updates；
- policy sampling RNG 与 observation/action noise RNG 分离；
- 配对 arms 不允许因为 noise sampling 消耗 policy RNG；
- observation/action/reward/success 的语义保持与 EXP-003 相同。

## 10. 预注册 checkpoint 分析

必须提前冻结两个分析点，不允许根据结果挑最好 checkpoint：

```text
Primary endpoint: update 100
Secondary endpoint: update 150
```

update 100 用于和 EXP-003 C4 的相同预算结果直接比较；update 150 用于观察更长适应是否带来收益或继续遗忘。

update 10–90、110–140 的周期 checkpoint 仅用于训练诊断。即使某个中间 checkpoint 最好，也不能替代主结果或次结果。

## 11. 正式评价设计

固定 evaluation seeds：

```text
20260920
20260921
```

每个 cell：

```text
episodes: 256
num_envs: 64
deterministic actor mean
reconfiguration_freq: 1
```

共同 evaluation 条件：

```text
clean / combined 0x
combined 1x
combined 2x
combined 3x
combined 4x
```

Primary evaluation 必须先完成：

```text
C0/C1/C2/C3 update-100 + EXP-003 C4 update-100
× seeds 1001/1002/1003
× clean 和 combined-4x
× evaluation seeds 20260920/20260921
= 60 cells
```

Secondary endpoint：

```text
C0/C1/C2/C3 update-150
× 3 training seeds
× clean 和 combined-4x
× 2 evaluation seeds
= 48 cells
```

Cross-severity secondary matrix 在时间允许时完成：C0–C3 的 update-100/update-150，以及 EXP-003 C4 update-100，在 1x/2x/3x evaluation 下的统一评价。原始 EXP-002 clean checkpoints 可作为 secondary reference；缺失的 3x original-policy cells 应用相同 evaluator 补齐。

如果总时限接近 10 小时，优先保证 60 个 primary cells、48 个 secondary endpoint cells 和 summary 完成，再运行 cross-severity 扩展。不得因为时间不足留下没有汇总的半套主结果。

记录：

- `success_once`；
- `success_at_end`；
- return；
- episode length；
- observation-noise magnitude；
- action-noise magnitude；
- noise-induced action clip fraction；
- source checkpoint、training seed、evaluation seed、multiplier、checkpoint update；
- wall-clock time、SPS、peak VRAM；
- 实际配置快照和代码 commit。

## 12. 主指标与成功门槛

对每个 multiplier `s ∈ {1,2,3}`，在共同 combined-4x evaluation 下计算：

```text
robustness_gain_s = success(C_s) - success(C0)
clean_cost_s      = success_clean(C_s) - success_clean(C0)
```

某个 multiplier 只有同时满足以下条件才算通过：

```text
mean robustness_gain_s >= +5 percentage points
至少 3 个 training seeds 中 2 个 robustness_gain_s > 0
mean clean_cost_s >= -5 percentage points
noisy return 不低于 matched C0
没有异常 action clipping 或其他退化机制
```

必须同时报告所有 arms 和所有 seeds，不能只报告最优 multiplier。多臂比较属于预注册 dose-response，不进行事后删除。

## 13. 结果解释规则

- 如果一个或多个温和 multiplier 通过：下一实验只对最有希望的预注册强度做 observation-only/action-only component ablation；
- 如果 noisy success 上升但 clean cost 超过 -5 pp：说明存在 robustness/forgetting trade-off，下一步考虑 KL-to-source policy 或 demonstration replay，而不是继续盲调 multiplier；
- 如果 1x–3x 全部失败：停止 interface-noise PPO multiplier 路线，转入 ACT/BC baseline；
- 如果只有单 seed 为正：不能称为方法有效；
- 如果 return 上升但 success 不升：检查 reward shaping、失败类型和 reward hacking；
- 如果 action clipping 异常：结果不能解释为噪声正则化收益；
- 不能把额外训练时间本身当成 noise treatment 收益，所有比较必须对 matched C0；
- 不能把 C4 的 update-100 结果与其他 arms 的 update-150 直接作为主比较。

## 14. GPU 调度与预计时长

每个 100-update EXP-003 run 实测约 1.54–1.58 小时。150 updates 预计约 2.3–2.4 小时。

建议按 training seed 分三波，每波四个 arms，并轮换 GPU，减少 arm 与 GPU 固定绑定：

```text
Wave seed1001: C0→GPU0, C1→GPU1, C2→GPU2, C3→GPU3
Wave seed1002: C0→GPU1, C1→GPU2, C2→GPU3, C3→GPU0
Wave seed1003: C0→GPU2, C1→GPU3, C2→GPU0, C3→GPU1
```

预计：

```text
formal training: 约 7.0–7.3 小时
primary + secondary evaluation: 约 1.0 小时
cross-severity evaluation、summary 和冗余: 约 0.5–1.5 小时
total: 约 8.5–9.8 小时
```

每个训练 subprocess 建议有不低于 4 小时但明确的 timeout；总 pipeline 建议有约 10 小时的外层 timeout，并保证 primary evaluation 在扩展评价之前执行。

## 15. 工程实现原则

优先复用 EXP-003 已验证的：

- checkpoint 完整恢复；
- field-aware noise；
- PPO continuation；
- periodic checkpoint；
- deterministic evaluation；
- launcher/manifest/summary 模式；
- smoke test 和 resume 安全检查。

但必须先审计，不能假设现有代码已经支持：

- 任意 arm label；
- 0x/1x/2x/3x 多臂 launcher；
- 固定 `warmup_updates=20`；
- 150 updates；
- update-100 与 update-150 双 endpoint；
- 统一 0x–4x evaluation grid；
- EXP-003 C4 reference checkpoint；
- primary-first evaluation 顺序；
- 完整矩阵验证和多臂 summary。

建议在 `exp004/` 下新增独立配置、launcher、summary 和测试，尽量调用 EXP-003 的稳定底层函数，避免复制整份 PPO。任何必须修改 `exp003/` 的通用化改动都要保持 EXP-003 行为和历史结果不变，并补回归测试。

建议文件框架，需在审计后确认：

```text
exp004/
├── NEW_CHAT_PROMPT.md
├── EXPERIMENT_PLAN.md
├── README.md
├── __init__.py
├── config.py
├── configs/
│   └── experiment.json
├── launch_smoke.py
├── launch_training.py
├── launch_evaluation.py
├── summarize.py
└── tests/
    ├── test_config_schedule.py
    ├── test_matrix.py
    └── test_summary.py
```

建议服务器输出：

```text
/share/yangpengju-local/embodied/runs/exp004/
```

建议本地私有结果：

```text
exp004/results/server_YYYYMMDD/
```

必须在写结果前把 `exp004/results/` 加入 `.gitignore`。

## 16. 测试与 smoke 计划

本地测试至少覆盖：

- multiplier 只允许 `{0,1,2,3}`，C4 为外部 reference；
- 12-run training matrix 无缺失、无重复；
- source seed/arm/checkpoint 映射正确；
- warmup 在 update 20 精确到达目标；
- update-100 和 update-150 global step 正确；
- GPU 轮换表正确；
- primary 60-cell 和 secondary 48-cell matrix 正确；
- summary 拒绝缺失、重复、错误 seed、错误 multiplier 或错误 checkpoint endpoint；
- success gate 使用 matched C0，而不是 original policy 或其他 multiplier；
- EXP-003 C4 只能进入 update-100 reference；
- JSON 配置与 manifest 可序列化且路径清楚。

服务器只允许一次合并 smoke：

```text
source seed: 1002
arms: C0/C1/C2/C3，各占一张 GPU
num_envs: 16
num_steps: 5
executed updates: 2
保存并重载 checkpoint
对 primary clean/4x 路径做少量 episode evaluation
检查 summary 能识别 smoke 为非正式数据
```

smoke 失败就停止，不得自动进入正式训练。正式 pipeline 必须使用单独输出目录，不能混入 smoke。

## 17. 失败恢复与数据安全

- 每个 run 独立目录、日志和 config snapshot；
- 只在完整 `final_ckpt.pt`、config、manifest 一致时允许 `--resume` 跳过；
- 非空但不完整目录必须人工审查，不能自动覆盖；
- 一个 wave 出现 failure 后不启动下一 wave；
- 训练完成后先验证 checkpoint keys、global step、optimizer state，再评价；
- 评价完成后验证 cell 数和 episodes 数，再 summary；
- 保留所有失败日志；
- 不删除 EXP-002/003 的任何文件；
- 不将 server results 或 checkpoints 提交 GitHub。

## 18. 完整实验结束后的必需汇报

EXP-004 完成后必须更新 `EXPERIMENT_RECORD.md`，并解释：

1. 研究问题和预注册假设；
2. 实际改变的唯一变量；
3. 固定变量；
4. 数据来自当前策略 rollout，为什么属于 on-policy PPO；
5. observation、action、reward、done 的形状和物理含义；
6. PPO loss 每一项；
7. checkpoint lineage、恢复的状态和继续训练的参数；
8. update-100/update-150、clean/OOD、逐 seed 和 aggregate 指标；
9. action clipping、return、失败案例和混杂因素；
10. 是否支持假设，以及下一步只改变什么。

同时主动带我回答实验理解自检七问：

1. observation、action、reward、done 是什么？
2. 数据来自当前策略、历史策略还是专家？
3. loss 的每一项优化什么？
4. checkpoint 从哪里来，哪些参数继续训练？
5. 唯一改变变量是什么？
6. 指标变化是否可能来自 seed、泄漏、评价设置或 reward hacking？
7. 代码改动是否真的符合设计？

## 19. 新对话的第一项任务

开始时不要执行服务器命令。请先：

1. 完整阅读第 4 节列出的文件；
2. 检查本地 Git 状态和 `.gitignore`；
3. 用自己的话总结 EXP-002、EXP-003 和 EXP-004 的因果区别；
4. 审计 EXP-003 代码哪些可以直接复用，哪些需要最小通用化；
5. 检查本预注册矩阵、endpoint、统计门槛和 8–10 小时时间预算是否自洽；
6. 给出拟新增/修改文件列表；
7. 给出本地测试计划；
8. 给出服务器 smoke、formal training、primary evaluation、secondary evaluation、summary 的执行顺序；
9. 明确列出任何你认为必须在写代码前修正的设计风险；
10. 停下来等待我审核。

不要在第一轮自动写大量代码、连接服务器或启动实验。不要因为今晚有 GPU 时间就牺牲实验可解释性。
