#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import CHECKPOINT, MaterializedDataset, collator, forward_actions, load_policy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--max-samples", type=int, default=0); args = ap.parse_args()
    vla, processor, head, projector = load_policy(Path(CHECKPOINT))
    vla.eval(); head.eval(); projector.eval()
    ds = MaterializedDataset("libero_spatial_no_noops", "train", processor, use_proprio=True)
    coll = collator(processor); preds = []; ids = []; start = time.time()
    for i in range(len(ds)):
        if args.max_samples and i >= args.max_samples: break
        item = ds[i]; pred = forward_actions(vla, head, projector, coll([item]), train_mode=False)[0].float().cpu().numpy()
        preds.append(pred); ids.append({"sample_index": int(i), "episode_id": str(item["sample_episode_id"]), "timestep": int(item["sample_timestep"])})
        if (i + 1) % 25 == 0: print(f"cached {i + 1}/{len(ds)}", flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); np.save(out.with_suffix(".npy"), np.stack(preds).astype(np.float32)); out.with_suffix(".json").write_text(json.dumps({"checkpoint": str(CHECKPOINT), "count": len(preds), "ids": ids, "elapsed_sec": time.time() - start}, indent=2))
    print(json.dumps({"count": len(preds), "elapsed_sec": time.time() - start, "output": str(out)}))


if __name__ == "__main__": main()
