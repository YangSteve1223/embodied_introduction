#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import CHECKPOINT, build_dataset, collator, forward_actions, forward_head_only, load_policy, parameter_snapshot


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_state(run_dir, step, head, projector, optimizer, scheduler, cfg):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_dir / f"action_head_step_{step}.pt")
    torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, run_dir / f"proprio_projector_step_{step}.pt")
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": cfg}, run_dir / f"optimizer_step_{step}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "head_only"], default="full")
    ap.add_argument("--suites", default="libero_spatial_no_noops,libero_object_no_noops")
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--save-freq", type=int, default=250)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    set_seed(args.seed)
    suites = tuple(s for s in args.suites.split(",") if s)
    vla, processor, head, projector = load_policy(Path(args.checkpoint))
    vla.eval(); projector.eval(); head.train()
    vla_probe = {name: p.detach().float().cpu().clone() for name, p in list(vla.named_parameters())[:3]}
    head_before, projector_before = parameter_snapshot(head), parameter_snapshot(projector)
    vla.requires_grad_(False); head.requires_grad_(True); projector.requires_grad_(args.mode == "full")
    trainable = list(head.parameters()) + (list(projector.parameters()) if args.mode == "full" else [])
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=min(50, max(1, args.max_steps)))
    dataset = build_dataset(processor, suites, "train", use_proprio=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator(processor), num_workers=0)
    iterator = iter(loader)
    losses, start = [], time.time()
    try:
        for step in range(1, args.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(args.grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader); batch = next(iterator)
                if args.mode == "full":
                    pred = forward_actions(vla, head, projector, batch, train_mode=True)
                else:
                    pred = forward_head_only(vla, head, projector, batch)
                gt = batch["actions"].to(pred.device, dtype=pred.dtype)
                loss = F.l1_loss(pred, gt) / args.grad_accum
                loss.backward()
                accum_loss += float(loss.detach().cpu()) * args.grad_accum
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
            optimizer.step(); scheduler.step()
            losses.append({"step": step, "train_l1": accum_loss / args.grad_accum, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0]})
            print(losses[-1], flush=True)
            if step % args.save_freq == 0:
                save_state(run_dir, step, head, projector, optimizer, scheduler, vars(args))
    except Exception as exc:
        (run_dir / "FAILED").write_text(repr(exc))
        (run_dir / "traceback.txt").write_text(__import__("traceback").format_exc())
        raise
    save_state(run_dir, args.max_steps, head, projector, optimizer, scheduler, vars(args))
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_dir / "action_head_final.pt")
    torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, run_dir / "proprio_projector_final.pt")
    (run_dir / "train_metrics.json").write_text(json.dumps({"losses": losses, "elapsed_sec": time.time() - start}, indent=2))
    deltas = {"head_max_abs_delta": max(float((parameter_snapshot(head)[k] - head_before[k]).abs().max()) for k in head_before),
              "projector_max_abs_delta": max(float((parameter_snapshot(projector)[k] - projector_before[k]).abs().max()) for k in projector_before)}
    vla_after = {name: p.detach().float().cpu() for name, p in list(vla.named_parameters())[:3]}
    deltas["backbone_probe_max_abs_delta"] = max(float((vla_after[k] - vla_probe[k]).abs().max()) for k in vla_probe)
    (run_dir / "parameter_deltas.json").write_text(json.dumps(deltas, indent=2))
    (run_dir / "SUCCESS").write_text(str(args.max_steps))
    print(json.dumps({"run_dir": str(run_dir), "mode": args.mode, "steps": args.max_steps, **deltas}, sort_keys=True))


if __name__ == "__main__":
    main()
