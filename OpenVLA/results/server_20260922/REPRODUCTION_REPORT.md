# OpenVLA-OFT offline post-training report

Date: 2026-09-22  
Server: `act-server` / `eva12-yangpengju`  
Run root: `/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260922`

## Outcome

Stage 0 audit, dataset acquisition, fixed episode split, offline baseline, head+proprio SFT, teacher-consistency distillation, and a LoRA smoke test completed. The primary head+proprio path completed a 1000-step formal run and its component checkpoints reload successfully. Selection used only the fixed held-out set, with all candidates retained.

## Protocol

- Dataset: `openvla/modified_libero_rlds`, revision `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`; downloaded via `https://hf-mirror.com` because `huggingface.co` was unreachable.
- Seed `20260922`; per-task episode-level 90/10 split. Spatial/Object only: 20 tasks, 1280 train and 640 held-out transitions (64/32 per task caps).
- Metric: normalized action chunk L1, plus first-step L1 and per-dimension MAE.

## Held-out results

| run | macro suite L1 | Spatial | Object | relative vs baseline |
|---|---:|---:|---:|---:|
| Official baseline | 0.164500 | 0.051142 | 0.277858 | — |
| Head+proprio SFT (step 250) | 0.137058 | 0.065988 | 0.208129 | 16.68% |
| Head+proprio distill (step 250) | 0.135026 | 0.063058 | 0.206993 | 17.92% |

SFT improves aggregate held-out L1 by 16.68% and Object, while Spatial worsens (forgetting signal). Distillation is slightly better in aggregate and retains Spatial somewhat better, but remains above baseline Spatial error.

## Reproduction and integrity

- SFT: 1000 steps, 3455.3 s; frozen backbone probe delta 0.0; head/projector deltas 0.0357666015625 / 0.035125732421875.
- Distillation: 500 steps, 2073.4 s; teacher cache has 640 Spatial train predictions.
- LoRA: rank-8 all-linear 3-step smoke succeeded; no formal claim.
- Final SFT reload smoke succeeded on one held-out sample; see `03_head_proprio_sft/formal_full_1000/reload_smoke.json`.
- No LIBERO/MuJoCo rendering or closed-loop success was run by design.
- The first distillation attempt hit a bf16 `smooth_l1_loss` limitation; retention loss was evaluated in float32 and the formal run completed. Original logs are retained.
