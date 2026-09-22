# OpenVLA-OFT offline post-training report

The complete server-side report is [here](<./results/server_20260922/REPRODUCTION_REPORT.md>).

Headline result on the fixed 640-sample Spatial+Object held-out set:

| run | macro normalized action L1 |
|---|---:|
| Official baseline | 0.164500 |
| Head+proprio SFT, selected step 250 | 0.137058 |
| Head+proprio distillation, selected step 250 | 0.135026 |

The 1000-step SFT and 500-step distillation formal runs completed, and the final SFT components reload successfully. Spatial error increased relative to baseline, so the aggregate improvement should not be interpreted as universal suite improvement. LoRA was only a 3-step smoke test.

## Independent small-student distillation follow-up

At the user's request, a true teacher-to-student experiment was then run separately. The official OpenVLA-7B teacher supplied cached Spatial action chunks; an independent 213,656-parameter CNN + language bag-of-words + proprio MLP student was trained with 3-GPU DDP for 1000 steps. Its held-out macro L1 was 0.213461 (Spatial 0.207281, Object 0.219642), worse than the 0.164500 teacher baseline, but this is a genuine standalone student result rather than an action-head update. Artifacts are in [true_distill](<./results/server_20260922/true_distill/>).

## 2.04B student follow-up

The small student was then replaced by a four-layer truncated OpenVLA student with 2,042,439,111 trainable parameters. Three RTX 3090 GPUs used about 22.9GB each. An eight-layer version was about 2.85B and OOM'd during AdamW state allocation. The 2.04B student reached held-out macro L1 0.165353 at step 500 (Spatial 0.167271, Object 0.163435), close to the 7B baseline 0.164500. Continuing to step 1000 did not improve the aggregate score: macro L1 0.170499 (Spatial 0.167914, Object 0.173084). Therefore step 500 is the selected checkpoint for this run. Lightweight logs and metrics are in `results/server_20260922/true_distill_2b/`; multi-gigabyte checkpoints remain on the server.
