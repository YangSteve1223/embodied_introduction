#!/usr/bin/env python3
"""Shared data, checkpoint, and forward utilities for offline experiments."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform

ROOT = Path(os.environ.get("REPRO_ROOT", "/share/yangpengju-local/openvla-oft-repro"))
CHECKPOINT = Path(os.environ.get("CHECKPOINT", str(ROOT / "checkpoints/openvla-7b-oft-finetuned-libero-spatial")))
MATERIALIZED = ROOT / "runs/post_training_20260922/01_dataset/materialized"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def register_openvla():
    for fn, args in [
        (AutoConfig.register, ("openvla", OpenVLAConfig)),
        (AutoImageProcessor.register, (OpenVLAConfig, PrismaticImageProcessor)),
        (AutoProcessor.register, (OpenVLAConfig, PrismaticProcessor)),
        (AutoModelForVision2Seq.register, (OpenVLAConfig, OpenVLAForActionPrediction)),
    ]:
        try:
            fn(*args)
        except ValueError:
            pass


def strip_ddp(state):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_component(path):
    return strip_ddp(torch.load(path, map_location="cpu", weights_only=True))


def load_policy(checkpoint_path=CHECKPOINT, device=DEVICE):
    register_openvla()
    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    stats_path = checkpoint_path / "dataset_statistics.json"
    if stats_path.exists():
        vla.norm_stats = json.loads(stats_path.read_text())
    head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
    head.load_state_dict(load_component(str(checkpoint_path / "action_head--150000_checkpoint.pt")))
    projector.load_state_dict(load_component(str(checkpoint_path / "proprio_projector--150000_checkpoint.pt")))
    return vla, processor, head.to(device=device, dtype=torch.bfloat16), projector.to(device=device, dtype=torch.bfloat16)


class MaterializedDataset(Dataset):
    def __init__(self, suite: str, split: str, processor, use_proprio=True):
        base = MATERIALIZED / suite
        self.primary = np.load(base / f"{split}_primary.npy", allow_pickle=False)
        self.wrist = np.load(base / f"{split}_wrist.npy", allow_pickle=False)
        self.actions = np.load(base / f"{split}_actions.npy", allow_pickle=False)
        self.proprio = np.load(base / f"{split}_proprio.npy", allow_pickle=False)
        self.language = np.load(base / f"{split}_language.npy", allow_pickle=True)
        self.task = np.load(base / f"{split}_task.npy", allow_pickle=True)
        self.episode_id = np.load(base / f"{split}_episode_id.npy", allow_pickle=True)
        self.timestep = np.load(base / f"{split}_timestep.npy", allow_pickle=False)
        self.suite = suite
        self.transform = RLDSBatchTransform(
            ActionTokenizer(processor.tokenizer), processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder, use_wrist_image=True, use_proprio=use_proprio,
        )

    def __len__(self):
        return int(self.actions.shape[0])

    def __getitem__(self, idx):
        lang = str(self.language[idx])
        sample = {
            "dataset_name": self.suite, "action": self.actions[idx],
            "observation": {"image_primary": self.primary[idx:idx + 1], "image_wrist": self.wrist[idx:idx + 1], "proprio": self.proprio[idx]},
            "task": {"language_instruction": lang.encode("utf-8")},
        }
        out = self.transform(sample)
        out["sample_task"] = str(self.task[idx])
        out["sample_episode_id"] = str(self.episode_id[idx])
        out["sample_timestep"] = int(self.timestep[idx])
        out["sample_index"] = int(idx)
        return out


def build_dataset(processor, suites, split, use_proprio=True):
    return ConcatDataset([MaterializedDataset(s, split, processor, use_proprio=use_proprio) for s in suites])


def collator(processor):
    return PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id)


def forward_actions(vla, action_head, proprio_projector, batch, train_mode=False):
    device = next(vla.parameters()).device
    labels = batch["labels"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixels = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device=device, dtype=torch.bfloat16)
    context = torch.enable_grad() if train_mode else torch.inference_mode()
    with context:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = vla(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixels, labels=labels,
                         output_hidden_states=True, proprio=proprio, proprio_projector=proprio_projector, use_film=False)
            token_ids = labels[:, 1:]
            mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
            num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1
            text_hidden = output.hidden_states[-1][:, num_patches:-1]
            bsz = input_ids.shape[0]
            action_hidden = text_hidden[mask].reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
            pred = action_head.predict_action(action_hidden)
    return pred


def forward_head_only(vla, action_head, proprio_projector, batch):
    """Cache frozen VLA features and backpropagate only through the action head."""
    device = next(vla.parameters()).device
    labels = batch["labels"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixels = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = vla(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixels, labels=labels,
                         output_hidden_states=True, proprio=proprio, proprio_projector=proprio_projector, use_film=False)
            token_ids = labels[:, 1:]
            mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
            num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1
            text_hidden = output.hidden_states[-1][:, num_patches:-1]
            bsz = input_ids.shape[0]
            hidden = text_hidden[mask].reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
    return action_head.predict_action(hidden.detach())


def parameter_snapshot(module):
    return {name: tensor.detach().float().cpu().clone() for name, tensor in module.named_parameters()}
