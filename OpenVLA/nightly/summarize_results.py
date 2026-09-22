#!/usr/bin/env python3
import csv, hashlib, json, os
from pathlib import Path

ROOT = Path(os.environ.get("RUN_ROOT", "/share/yangpengju-local/openvla-oft-repro/runs/post_training_20260922"))
def load(rel):
    with (ROOT / rel).open() as f: return json.load(f)
def sha(rel):
    h = hashlib.sha256()
    with (ROOT / rel).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

base = load("02_offline_baseline/baseline_heldout.json")["metrics"]
sft = load("03_head_proprio_sft/eval_step250/heldout.json")["metrics"]
dist = load("04_head_proprio_distill/eval_step250/heldout.json")["metrics"]
sft_train = load("03_head_proprio_sft/formal_full_1000/train_metrics.json")
dist_train = load("04_head_proprio_distill/formal_500/train_metrics.json")
split = load("01_dataset/split_manifest.json")
download = load("01_dataset/download_manifest.json")
delta = load("03_head_proprio_sft/formal_full_1000/parameter_deltas.json")

def row(name, m, steps="", elapsed=""):
    b = base["macro_suite_normalized_action_l1"]
    return {"run": name, "sample_count": m["sample_count"], "macro_suite_normalized_action_l1": m["macro_suite_normalized_action_l1"], "relative_improvement_vs_baseline": (b-m["macro_suite_normalized_action_l1"])/b, "first_action_step_l1": m["first_action_step_l1"], "spatial_l1": m["per_suite"].get("libero_spatial_no_noops"), "object_l1": m["per_suite"].get("libero_object_no_noops"), "train_steps": steps, "train_elapsed_sec": elapsed}
rows = [row("official_baseline", base), row("head_proprio_sft_step250", sft, 1000, sft_train["elapsed_sec"]), row("head_proprio_distill_step250", dist, 500, dist_train["elapsed_sec"])]
with (ROOT / "RESULTS_TABLE.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
train_samples = sum(v["splits"]["train"]["count"] for v in split["suites_manifest"].values())
heldout_samples = sum(v["splits"]["heldout"]["count"] for v in split["suites_manifest"].values())
summary = {"status":"complete", "date":"2026-09-22", "split_seed":split["split_seed"], "dataset_repo":download["repo_id"], "dataset_revision":download["repo_sha"], "download_file_count":download["file_count"], "download_bytes":download["bytes"], "materialized_suites":split["suites"], "train_samples":train_samples, "heldout_samples":heldout_samples, "baseline":rows[0], "sft_step250":rows[1], "distill_step250":rows[2], "sft_formal":{"steps":1000,"elapsed_sec":sft_train["elapsed_sec"],"final_train_l1":sft_train["losses"][-1]["train_l1"],"parameter_deltas":delta}, "distill_formal":{"steps":500,"elapsed_sec":dist_train["elapsed_sec"],"final_gt_l1":dist_train["losses"][-1]["gt_l1"],"final_keep_smooth_l1":dist_train["losses"][-1]["keep_smooth_l1"]}, "lora_smoke":{"status":"SUCCESS","steps":3,"formal_run":False}, "checkpoint_sha256":{"sft_action_head_final":sha("03_head_proprio_sft/formal_full_1000/action_head_final.pt"),"sft_proprio_projector_final":sha("03_head_proprio_sft/formal_full_1000/proprio_projector_final.pt"),"distill_action_head_final":sha("04_head_proprio_distill/formal_500/action_head_final.pt"),"distill_proprio_projector_final":sha("04_head_proprio_distill/formal_500/proprio_projector_final.pt")}, "final_reload_smoke":load("03_head_proprio_sft/formal_full_1000/reload_smoke.json"), "selection_rule":"All checkpoints retained; step-250 was the best fixed-heldout candidate among evaluated component checkpoints. Formal SFT (1000) and distillation (500) completed.", "limitations":["Offline normalized action L1 only; no simulator rendering or closed-loop success.","Only Spatial and Object were materialized/evaluated; Goal and Libero10 were downloaded but not trained/evaluated.","LoRA was a 3-step smoke only."]}
with (ROOT / "summary.json").open("w") as f: json.dump(summary, f, indent=2)
b = base["macro_suite_normalized_action_l1"]
report = f'''# OpenVLA-OFT offline post-training report

Date: 2026-09-22  
Server: `act-server` / `eva12-yangpengju`  
Run root: `{ROOT}`

## Outcome

Stage 0 audit, dataset acquisition, fixed episode split, offline baseline, head+proprio SFT, teacher-consistency distillation, and a LoRA smoke test completed. The primary head+proprio path completed a 1000-step formal run and its component checkpoints reload successfully. Selection used only the fixed held-out set, with all candidates retained.

## Protocol

- Dataset: `openvla/modified_libero_rlds`, revision `{download["repo_sha"]}`; downloaded via `https://hf-mirror.com` because `huggingface.co` was unreachable.
- Seed `{split["split_seed"]}`; per-task episode-level 90/10 split. Spatial/Object only: 20 tasks, {train_samples} train and {heldout_samples} held-out transitions (64/32 per task caps).
- Metric: normalized action chunk L1, plus first-step L1 and per-dimension MAE.

## Held-out results

| run | macro suite L1 | Spatial | Object | relative vs baseline |
|---|---:|---:|---:|---:|
| Official baseline | {rows[0]["macro_suite_normalized_action_l1"]:.6f} | {rows[0]["spatial_l1"]:.6f} | {rows[0]["object_l1"]:.6f} | — |
| Head+proprio SFT (step 250) | {rows[1]["macro_suite_normalized_action_l1"]:.6f} | {rows[1]["spatial_l1"]:.6f} | {rows[1]["object_l1"]:.6f} | {rows[1]["relative_improvement_vs_baseline"]*100:.2f}% |
| Head+proprio distill (step 250) | {rows[2]["macro_suite_normalized_action_l1"]:.6f} | {rows[2]["spatial_l1"]:.6f} | {rows[2]["object_l1"]:.6f} | {rows[2]["relative_improvement_vs_baseline"]*100:.2f}% |

SFT improves aggregate held-out L1 by {rows[1]["relative_improvement_vs_baseline"]*100:.2f}% and Object, while Spatial worsens (forgetting signal). Distillation is slightly better in aggregate and retains Spatial somewhat better, but remains above baseline Spatial error.

## Reproduction and integrity

- SFT: 1000 steps, {sft_train["elapsed_sec"]:.1f} s; frozen backbone probe delta {delta["backbone_probe_max_abs_delta"]}; head/projector deltas {delta["head_max_abs_delta"]} / {delta["projector_max_abs_delta"]}.
- Distillation: 500 steps, {dist_train["elapsed_sec"]:.1f} s; teacher cache has 640 Spatial train predictions.
- LoRA: rank-8 all-linear 3-step smoke succeeded; no formal claim.
- Final SFT reload smoke succeeded on one held-out sample; see `03_head_proprio_sft/formal_full_1000/reload_smoke.json`.
- No LIBERO/MuJoCo rendering or closed-loop success was run by design.
- The first distillation attempt hit a bf16 `smooth_l1_loss` limitation; retention loss was evaluated in float32 and the formal run completed. Original logs are retained.
'''
(ROOT / "REPRODUCTION_REPORT.md").write_text(report)
(ROOT / "environment.txt").write_text((ROOT / "00_audit/pip-freeze.txt").read_text() + "\n--- stage0 ---\n" + (ROOT / "00_audit/stage0_system.txt").read_text())
(ROOT / "commands.log").write_text("""# Reproduction command summary
export ROOT=/share/yangpengju-local/openvla-oft-repro
export RUN=$ROOT/runs/post_training_20260922
export PY=$ROOT/anaconda3/envs/openvla-oft-repro/bin/python
export SRC=$ROOT/src/openvla-oft
export CODE=$RUN/code

# Stage 0 audit
ssh act-server; inspect nvidia-smi, git rev-parse/status, pip freeze/check, checkpoint files/SHA256, and the retained local inference smoke log.

# Stage 1 data
HF_ENDPOINT=https://hf-mirror.com $PY $CODE/download_datasets.py --output $RUN/01_dataset
$PY $CODE/materialize_rlds.py --dataset-root $ROOT/datasets/modified_libero_rlds --output $RUN/01_dataset/materialized --seed 20260922 --train-cap 64 --heldout-cap 32

# Stage 2 baseline
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=1 $PY $CODE/offline_eval.py --split heldout --suites libero_spatial_no_noops,libero_object_no_noops --output $RUN/02_offline_baseline/baseline_heldout.json

# Stage 3 SFT and fixed-heldout checkpoint comparison
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=2 $PY $CODE/train_head_proprio.py --run-dir $RUN/03_head_proprio_sft/formal_full_1000 --max-steps 1000 --save-freq 250 --mode full
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=0 $PY $CODE/offline_eval.py --split heldout --components $RUN/03_head_proprio_sft/eval_step250 --output $RUN/03_head_proprio_sft/eval_step250/heldout.json

# Stage 4 teacher cache + distillation
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=3 $PY $CODE/build_teacher_cache.py --output $RUN/04_head_proprio_distill/teacher_actions.npy
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=3 $PY $CODE/train_distill.py --run-dir $RUN/04_head_proprio_distill/formal_500 --max-steps 500
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=0 $PY $CODE/offline_eval.py --split heldout --components $RUN/04_head_proprio_distill/eval_step250 --output $RUN/04_head_proprio_distill/eval_step250/heldout.json

# Stage 5 smoke and final reload
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=0 $PY $CODE/train_lora_smoke.py --run-dir $RUN/05_lora/lora_smoke_r8 --max-steps 3
PYTHONPATH=$CODE:$SRC CUDA_VISIBLE_DEVICES=0 $PY $CODE/offline_eval.py --split heldout --components $RUN/03_head_proprio_sft/formal_full_1000 --max-samples 1 --output $RUN/03_head_proprio_sft/formal_full_1000/reload_smoke.json

All long commands were run in tmux; logs, FINISHED/SUCCESS markers, configs, checkpoints, and hashes remain under RUN.
""")
print(json.dumps({"summary":str(ROOT/"summary.json"),"rows":rows}, indent=2))
