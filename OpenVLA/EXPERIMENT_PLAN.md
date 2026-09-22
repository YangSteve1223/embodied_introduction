# OpenVLA-OFT offline post-training plan

This run follows `NEW_CHAT_PROMPT.md` and the server manual.

1. Stage 0: audit the server, repository commits, environment, checkpoint integrity, GPU/VRAM, and local inference smoke.
2. Stage 1: download the official `openvla/modified_libero_rlds` snapshot, materialize deterministic episode-level 90/10 train/held-out splits, and record manifests.
3. Stage 2: establish the official-checkpoint offline baseline on the fixed held-out set.
4. Stage 3: train the action head plus proprio projector with a frozen VLA backbone, retaining periodic checkpoints and selecting by held-out normalized action L1.
5. Stage 4: cache teacher actions and run head+proprio consistency distillation; compare on the same held-out set.
6. Stage 5: smoke-test rank-8 LoRA. A formal LoRA claim requires a separate longer run and evaluation.

The protocol intentionally omits LIBERO/MuJoCo rendering and closed-loop success. All candidates, logs, hashes, manifests, and environment captures are retained on the server under `runs/post_training_20260922/`; lightweight summaries are copied into `results/server_20260922/`.
