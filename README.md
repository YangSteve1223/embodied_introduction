# Embodied Introduction

This repository records a guided study of embodied-intelligence post-training.

The current project has two layers:

- `EXPERIMENT_CONTRACT.md`: the design contract used before running an experiment;
- `EXPERIMENT_RECORD.md`: the evidence record for completed and planned experiments.

The first completed experiment is a state-based PPO baseline in ManiSkill:

```text
PickCube-v1 + Franka Panda + GPU PhysX + headless state observations
```

The next planned experiment studies robustness on `PegInsertionSide-v1` under controlled observation/action noise. The NVIDIA graphics runtime on the shared server is not currently exposed, so state-only GPU simulation is used for high-throughput experiments and rendering is treated as a separate qualitative-evaluation concern.

## Repository policy

Source documents, experiment contracts, records, and analysis notes belong in Git. Generated logs, checkpoints, datasets, videos, local environments, secrets, and server-specific outputs are excluded by `.gitignore`.

## Safety

Do not commit SSH private keys, passwords, API tokens, server credentials, or large generated experiment outputs. Follow the GPU server rules in `服务器用户手册.md` before running or modifying anything remotely.
