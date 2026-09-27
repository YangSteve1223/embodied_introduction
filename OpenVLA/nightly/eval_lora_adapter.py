#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from common import CHECKPOINT, build_dataset, collator, forward_actions, load_policy


def summarize(rows):
    by_task, by_suite = defaultdict(list), defaultdict(list)
    for row in rows:
        by_task[(row["suite"], row["task"])].append(row["full_chunk_l1"])
        by_suite[row["suite"]].append(row["full_chunk_l1"])
    task_metrics = {f"{s}::{t}": float(np.mean(v)) for (s, t), v in sorted(by_task.items())}
    suite_metrics = {s: float(np.mean(v)) for s, v in sorted(by_suite.items())}
    return {
        "sample_count": len(rows),
        "per_task": task_metrics,
        "per_suite": suite_metrics,
        "macro_task_normalized_action_l1": float(np.mean(list(task_metrics.values()))) if task_metrics else None,
        "macro_suite_normalized_action_l1": float(np.mean(list(suite_metrics.values()))) if suite_metrics else None,
        "first_action_step_l1": float(np.mean([r["first_action_l1"] for r in rows])),
        "per_dimension_mae": np.mean(np.asarray([r["per_dimension_mae"] for r in rows]), axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--suites", default="libero_spatial_no_noops,libero_object_no_noops")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    vla, processor, head, projector = load_policy(Path(args.checkpoint))
    vla = PeftModel.from_pretrained(vla, args.adapter)
    vla.eval(); head.eval(); projector.eval()
    suites = tuple(x for x in args.suites.split(",") if x)
    ds = build_dataset(processor, suites, args.split, use_proprio=True)
    coll = collator(processor)
    rows = []
    start = time.time()
    for idx in range(len(ds)):
        if args.max_samples and idx >= args.max_samples:
            break
        item = ds[idx]
        pred = forward_actions(vla, head, projector, coll([item]), train_mode=False)[0].float().cpu().numpy()
        gt = np.asarray(item["actions"], dtype=np.float32)
        err = np.abs(pred - gt)
        rows.append({
            "sample_index": idx,
            "suite": str(item["dataset_name"]),
            "task": str(item["sample_task"]),
            "episode_id": str(item["sample_episode_id"]),
            "timestep": int(item["sample_timestep"]),
            "full_chunk_l1": float(err.mean()),
            "first_action_l1": float(err[0].mean()),
            "per_dimension_mae": err.mean(axis=0).tolist(),
            "finite": bool(np.isfinite(pred).all()),
        })
        if (idx + 1) % 25 == 0:
            print(f"evaluated {idx + 1}/{len(ds)}", flush=True)
    result = {"config": vars(args), "elapsed_sec": time.time() - start,
              "metrics": summarize(rows), "rows": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(out), **result["metrics"],
                      "elapsed_sec": result["elapsed_sec"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
