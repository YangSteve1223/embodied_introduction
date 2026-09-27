#!/usr/bin/env python3
"""Offline AWR-style OpenVLA adaptation with Spatial retention.

This is the RL-compatible path for the server's documented no-rendering setup:
clean demonstrations are treated as positive-return actions, and the advantage
proxy is the frozen/base policy's normalized action error on each Object item.
The resulting weighted regression is deliberately reported as offline RL/AWR,
not as online PPO or a closed-loop success-rate experiment.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

from common import CHECKPOINT, build_dataset, collator, forward_actions, load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--retain-weight", type=float, default=8.0)
    ap.add_argument("--adv-temperature", type=float, default=0.05)
    ap.add_argument("--adv-clip", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260924)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))

    teacher_cache = np.load(args.teacher_cache, allow_pickle=False)
    vla, processor, head, projector = load_policy(Path(args.checkpoint))
    lora = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora)
    vla.train()
    head.eval(); projector.eval()
    head.requires_grad_(False); projector.requires_grad_(False)
    params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.0)

    object_ds = build_dataset(processor, ("libero_object_no_noops",), "train", use_proprio=True)
    spatial_ds = build_dataset(processor, ("libero_spatial_no_noops",), "train", use_proprio=True)
    coll = collator(processor)
    metrics = []
    start = time.time()
    running_mean = 0.20
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        object_item = object_ds[random.randrange(len(object_ds))]
        object_batch = coll([object_item])
        object_pred = forward_actions(vla, head, projector, object_batch, train_mode=True)
        object_gt = torch.from_numpy(np.asarray(object_item["actions"])).to(
            object_pred.device, dtype=object_pred.dtype).unsqueeze(0)
        base_error = float(F.l1_loss(object_pred.detach(), object_gt))
        running_mean = 0.98 * running_mean + 0.02 * base_error
        advantage = float(np.clip((base_error - running_mean) / args.adv_temperature,
                                  -args.adv_clip, args.adv_clip))
        weight = float(np.exp(advantage))
        object_per_elem = F.smooth_l1_loss(
            object_pred.float(), object_gt.float(), reduction="none"
        )
        object_loss = object_per_elem.mean()
        (object_loss * weight * 0.5).backward()

        spatial_item = spatial_ds[random.randrange(len(spatial_ds))]
        spatial_batch = coll([spatial_item])
        spatial_pred = forward_actions(vla, head, projector, spatial_batch, train_mode=True)
        teacher = torch.from_numpy(teacher_cache[int(spatial_item["sample_index"]) ]).to(
            spatial_pred.device, dtype=spatial_pred.dtype).unsqueeze(0)
        retention_loss = F.smooth_l1_loss(
            spatial_pred.float(), teacher.float()
        ).to(spatial_pred.dtype)
        (retention_loss * args.retain_weight * 0.5).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        row = {
            "step": step,
            "object_gt_l1": float(object_loss.detach()),
            "offline_advantage": advantage,
            "awr_weight": weight,
            "running_base_error": running_mean,
            "spatial_teacher_smooth_l1": float(retention_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
        }
        metrics.append(row)
        if step == 1 or step % 25 == 0:
            print(row, flush=True)

    adapter_dir = run / "adapter"
    vla.save_pretrained(adapter_dir)
    result = {
        "config": vars(args),
        "elapsed_sec": time.time() - start,
        "train_metrics": metrics,
        "adapter_dir": str(adapter_dir),
        "method": "offline_awr_style_lora",
        "reward_definition": "expert demonstration; advantage proxy is frozen/base policy Object action L1 error",
        "closed_loop_rl": False,
    }
    (run / "train_metrics.json").write_text(json.dumps(result, indent=2))
    (run / "SUCCESS").write_text(str(args.max_steps))
    print(json.dumps({"adapter_dir": str(adapter_dir), "steps": args.max_steps,
                      "elapsed_sec": result["elapsed_sec"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
