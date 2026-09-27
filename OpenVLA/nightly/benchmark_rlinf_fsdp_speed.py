#!/usr/bin/env python3
"""Measure OpenVLA post-training throughput with single-GPU vs RLinf-style FSDP.

The benchmark keeps the data, model, LoRA target modules, optimizer, and total
number of samples fixed.  Rank 0 prints a JSON record suitable for later
comparison.  This is intentionally a speed benchmark, not a quality run.
"""

from __future__ import annotations

import argparse
import json
import importlib.machinery
import math
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW

DEFAULT_CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "/share/yangpengju-local/openvla-oft-repro/checkpoints/openvla-7b-oft-finetuned-libero-spatial",
)

def init_dist(world_size: int) -> tuple[int, int, torch.device]:
    if world_size == 1:
        return 0, 1, torch.device("cuda:0")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # Some RLinf/Torch startup paths may initialize the default group while
    # importing distributed helpers. Reuse it if already present.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def maybe_wrap_fsdp(model: torch.nn.Module, device: torch.device, world_size: int):
    if world_size == 1:
        return model, "replicated_single_gpu"
    rlinf_site = os.environ.get("RLINF_SITE_PACKAGES")
    if rlinf_site and rlinf_site not in sys.path:
        # Append after the environment's native site-packages so OpenVLA keeps
        # its validated transformers/peft versions while RLinf-only packages
        # such as OmegaConf and Ray remain importable.
        sys.path.append(rlinf_site)

    from rlinf.hybrid_engines.fsdp import CPUOffload, FSDP
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import init_fn
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.scheduler import Worker
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    Worker.torch_platform.set_device(int(os.environ["LOCAL_RANK"]))

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    # This is the same low-peak initialization path used by RLinf's FSDP
    # strategy: keep rank 0's checkpoint on CPU, initialize non-zero ranks
    # with empty tensors, then broadcast/shard during FSDP construction.
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config={"wrap_policy": {}, "use_orig_params": True},
        is_lora=True,
        model_type="openvla_oft",
    )
    wrapped = FSDP(
        module=model,
        param_init_fn=init_fn,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        use_orig_params=True,
        device_id=int(os.environ["LOCAL_RANK"]),
        limit_all_gathers=True,
        sync_module_states=True,
        cpu_offload=CPUOffload(offload_params=False),
    )
    return wrapped, "rlinf_fsdp_full_shard"


def train(args: argparse.Namespace) -> dict:
    rank, world_size, device = init_dist(args.world_size)
    if torch.cuda.is_available() is not True:
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    # The RLinf venv owns the validated Torch/FSDP stack. OpenVLA's package
    # dependencies are appended after interpreter startup so they cannot
    # replace that Torch version.
    openvla_site = os.environ.get("OPENVLA_SITE_PACKAGES")
    if openvla_site:
        # Cache RLinf's validated torchvision build before exposing the
        # OpenVLA environment's pure-Python package set. Otherwise the latter
        # would shadow a torchvision extension compiled for Torch 2.2.
        import functorch  # noqa: F401
        import torchvision  # noqa: F401

        if openvla_site not in sys.path:
            sys.path.insert(0, openvla_site)

    # The benchmark does not log to Weights & Biases. timm imports wandb
    # opportunistically, while the two validated environments carry
    # incompatible protobuf/wandb combinations; keep that optional import
    # from changing the training stack.
    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.__version__ = "0.0-disabled"
        wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
        wandb_stub.init = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb_stub

    # Import PEFT only after the OpenVLA-compatible transformers path is
    # selected above.
    from peft import LoraConfig, get_peft_model

    # Importing prismatic before the NCCL process group exists can create a
    # default Gloo group. Delay the import until after init_dist().
    from common import build_dataset, collator, forward_actions, load_policy

    vla, processor, head, projector = load_policy(
        Path(args.checkpoint), device=torch.device("cpu")
    )
    if world_size == 1:
        vla = vla.to(device)
    head = head.to(device)
    projector = projector.to(device)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora)
    vla, parallel_mode = maybe_wrap_fsdp(vla, device, world_size)
    vla.train()
    head.eval(); projector.eval()
    head.requires_grad_(False); projector.requires_grad_(False)
    params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.0)

    ds = build_dataset(
        processor,
        tuple(x for x in args.suites.split(",") if x),
        "train",
        use_proprio=True,
    )
    coll = collator(processor)

    # Keep the amount of useful work fixed across world sizes.
    local_steps = math.ceil(args.global_samples / world_size)
    warmup_steps = args.warmup_steps
    total_steps = warmup_steps + local_steps

    def one_step(step: int):
        sample_idx = ((step - warmup_steps) * world_size + rank) % len(ds)
        batch = coll([ds[sample_idx]])
        optimizer.zero_grad(set_to_none=True)
        pred = forward_actions(vla, head, projector, batch, train_mode=True)
        gt = batch["actions"].to(pred.device, dtype=pred.dtype)
        loss = torch.nn.functional.l1_loss(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        return float(loss.detach().float().cpu())

    for step in range(warmup_steps):
        one_step(step)
    torch.cuda.synchronize(device)
    if world_size > 1:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    losses = []
    for step in range(warmup_steps, total_steps):
        losses.append(one_step(step))
    torch.cuda.synchronize(device)
    if world_size > 1:
        dist.barrier()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    record = {
        "world_size": world_size,
        "parallel_mode": parallel_mode,
        "global_samples": world_size * local_steps,
        "local_steps": local_steps,
        "warmup_steps": warmup_steps,
        "model": str(args.checkpoint),
        "suites": args.suites,
        "lora_rank": args.lora_rank,
        "lr": args.lr,
        "train_elapsed_sec": elapsed,
        "train_samples_per_sec": (world_size * local_steps) / elapsed,
        "train_sec_per_global_sample": elapsed / (world_size * local_steps),
        "train_sec_per_step": elapsed / local_steps,
        "peak_memory_gb_rank0_or_local": peak_gb,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "device": torch.cuda.get_device_name(device),
    }
    if rank == 0:
        print(json.dumps(record, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--suites", default="libero_spatial_no_noops,libero_object_no_noops")
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--global-samples", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    record = train(args)
    if int(os.environ.get("RANK", "0")) == 0:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
