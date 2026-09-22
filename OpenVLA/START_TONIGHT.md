# 给 Luna 的短 Prompt

请把工作目录切到 `/Users/yangda/Documents/TaskandWork/embodied_introduction/openvla`，先完整阅读同级 `../服务器用户手册.md` 和 `NEW_CHAT_PROMPT.md`，再严格按长 prompt 直接开始今晚的 OpenVLA-OFT 离线后训练。你已获授权通过 `ssh act-server` 在服务器规范内持续执行、监控和恢复约 10–12 小时，不要等待我逐步确认。按 `只读审计 → clean 多场景数据 → 固定 held-out baseline → head+proprio（必要时 head-only）→ Spatial teacher retention distillation → 可行时 LoRA → 汇总` 的顺序推进，至少完整闭环一项后训练。不要再处理 EGL，不加入噪声或视觉扰动。数据优先由服务器直接下载；只有服务器确实无法下载时才允许 Mac 中转，且 Mac 下载加上传总流量不得超过 10 GB。不要在 Mac 下载大模型或完整数据仓库。结束后只把必要的轻量结果拉到 `openvla/results/server_20260922/`，并完成报告。现在从 Stage 0 开始。
