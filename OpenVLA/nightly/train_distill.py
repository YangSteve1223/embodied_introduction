#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from common import CHECKPOINT, build_dataset, collator, forward_actions, load_policy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--suites", default="libero_spatial_no_noops,libero_object_no_noops"); ap.add_argument("--max-steps", type=int, default=500); ap.add_argument("--grad-accum", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=20260922); args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    cache = np.load(args.cache)
    vla, processor, head, projector = load_policy(Path(CHECKPOINT)); vla.eval(); head.train(); projector.train(); vla.requires_grad_(False); head.requires_grad_(True); projector.requires_grad_(True)
    optimizer = AdamW(list(head.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=0.0)
    dataset = build_dataset(processor, tuple(args.suites.split(",")), "train", use_proprio=True); coll = collator(processor)
    losses = []; start = time.time()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True); total = 0.0; keep_total = 0.0
        for _ in range(args.grad_accum):
            idx = random.randrange(len(dataset)); item = dataset[idx]; pred = forward_actions(vla, head, projector, coll([item]), train_mode=True)
            gt = torch.from_numpy(np.asarray(item["actions"])).to(pred.device, dtype=pred.dtype).unsqueeze(0)
            gt_loss = F.l1_loss(pred, gt); keep = torch.zeros((), device=pred.device, dtype=pred.dtype)
            if str(item["dataset_name"]) == "libero_spatial_no_noops":
                teacher = torch.from_numpy(cache[int(item["sample_index"]) ]).to(pred.device, dtype=pred.dtype).unsqueeze(0)
                keep = F.smooth_l1_loss(pred.float(), teacher.float()).to(pred.dtype)
            loss = (gt_loss + 0.25 * keep) / args.grad_accum; loss.backward(); total += float(gt_loss.detach()); keep_total += float(keep.detach())
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(projector.parameters()), 1.0); optimizer.step()
        row = {"step": step, "gt_l1": total / args.grad_accum, "keep_smooth_l1": keep_total / args.grad_accum}; losses.append(row); print(row, flush=True)
        if step % 250 == 0:
            torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_dir / f"action_head_step_{step}.pt")
            torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, run_dir / f"proprio_projector_step_{step}.pt")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_dir / "action_head_final.pt")
    torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, run_dir / "proprio_projector_final.pt")
    (run_dir / "train_metrics.json").write_text(json.dumps({"losses": losses, "elapsed_sec": time.time() - start}, indent=2)); (run_dir / "SUCCESS").write_text(str(args.max_steps))


if __name__ == "__main__": main()
