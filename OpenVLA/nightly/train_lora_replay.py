#!/usr/bin/env python3
"""Object-specialized OpenVLA-OFT LoRA training with Spatial replay retention.

The base checkpoint is already specialized on LIBERO Spatial.  Each update
uses one held-in Object demonstration for specialization and one Spatial
demonstration with a frozen-base teacher action target to reduce forgetting.
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
    ap.add_argument("--retain-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260923)
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
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        object_item = object_ds[random.randrange(len(object_ds))]
        object_batch = coll([object_item])
        object_pred = forward_actions(vla, head, projector, object_batch, train_mode=True)
        object_gt = torch.from_numpy(np.asarray(object_item["actions"])).to(
            object_pred.device, dtype=object_pred.dtype).unsqueeze(0)
        object_loss = F.l1_loss(object_pred, object_gt)
        (object_loss * 0.5).backward()

        spatial_item = spatial_ds[random.randrange(len(spatial_ds))]
        spatial_batch = coll([spatial_item])
        spatial_pred = forward_actions(vla, head, projector, spatial_batch, train_mode=True)
        teacher = torch.from_numpy(teacher_cache[int(spatial_item["sample_index"]) ]).to(
            spatial_pred.device, dtype=spatial_pred.dtype).unsqueeze(0)
        retention_loss = F.smooth_l1_loss(spatial_pred.float(), teacher.float()).to(spatial_pred.dtype)
        (retention_loss * args.retain_weight * 0.5).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        row = {
            "step": step,
            "object_gt_l1": float(object_loss.detach()),
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
    }
    (run / "train_metrics.json").write_text(json.dumps(result, indent=2))
    (run / "SUCCESS").write_text(str(args.max_steps))
    print(json.dumps({"adapter_dir": str(adapter_dir), "steps": args.max_steps,
                      "elapsed_sec": result["elapsed_sec"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
