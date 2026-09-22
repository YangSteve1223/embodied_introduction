#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import CHECKPOINT, build_dataset, collator, forward_actions, load_policy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--max-steps", type=int, default=3); args = ap.parse_args()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True); (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    vla, processor, head, projector = load_policy(Path(CHECKPOINT)); vla.vision_backbone.set_num_images_in_input(2)
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian")
    vla = get_peft_model(vla, lora); vla.train(); head.eval(); projector.eval(); head.requires_grad_(False); projector.requires_grad_(False)
    params = [p for p in vla.parameters() if p.requires_grad]; optimizer = AdamW(params, lr=5e-5)
    ds = build_dataset(processor, ("libero_spatial_no_noops", "libero_object_no_noops"), "train", use_proprio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collator(processor), num_workers=0); it = iter(loader)
    try:
        for step in range(1, args.max_steps + 1):
            try: batch = next(it)
            except StopIteration: it = iter(loader); batch = next(it)
            optimizer.zero_grad(set_to_none=True); pred = forward_actions(vla, head, projector, batch, train_mode=True); loss = torch.nn.functional.l1_loss(pred, batch["actions"].to(pred.device, dtype=pred.dtype)); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step(); print({"step": step, "loss": float(loss.detach())}, flush=True)
        vla.save_pretrained(run / "adapter"); (run / "SUCCESS").write_text(str(args.max_steps))
    except Exception as exc:
        (run / "FAILED").write_text(repr(exc)); (run / "traceback.txt").write_text(__import__("traceback").format_exc()); raise


if __name__ == "__main__": main()
