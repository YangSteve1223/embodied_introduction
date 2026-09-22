#!/usr/bin/env python3
"""Materialize a fixed episode-level RLDS train/heldout subset."""
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

ROOT = Path(os.environ.get("REPRO_ROOT", "/share/yangpengju-local/openvla-oft-repro"))
DATA_ROOT = ROOT / "datasets" / "modified_libero_rlds"
RUN_ROOT = ROOT / "runs" / "post_training_20260922"
OUT_ROOT = RUN_ROOT / "01_dataset" / "materialized"
SEED = int(os.environ.get("SPLIT_SEED", "20260922"))
SUITES = tuple(os.environ.get("MATERIALIZE_SUITES", "libero_spatial_no_noops,libero_object_no_noops").split(","))
TRAIN_PER_TASK = int(os.environ.get("TRAIN_PER_TASK", "64"))
HELDOUT_PER_TASK = int(os.environ.get("HELDOUT_PER_TASK", "32"))


def episode_id(meta: dict, index: int) -> str:
    raw = meta.get("file_path", b"")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return f"{raw}::episode_index={index}"


def task_from_language(language: bytes) -> str:
    if isinstance(language, bytes):
        return language.decode("utf-8", errors="replace").strip().lower()
    return str(language).strip().lower()


def raw_episode_dataset(name: str, skip_images: bool):
    builder = tfds.builder(name, data_dir=str(DATA_ROOT))
    if skip_images:
        skip = tfds.decode.SkipDecoding()
        decoders = {"steps": {"observation": {"image": skip, "wrist_image": skip}}}
        return builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    return builder.as_dataset(split="train", shuffle_files=False)


def transformed_arrays(episode: dict):
    steps = list(episode["steps"])
    actions = np.stack([step["action"] for step in steps]).astype(np.float32)
    # Official libero_dataset_transform: keep 6D EEF action and flip clipped gripper.
    actions = np.concatenate([actions[:, :6], 1.0 - np.clip(actions[:, -1:], 0.0, 1.0)], axis=1)
    proprio = np.stack([step["observation"]["state"] for step in steps]).astype(np.float32)
    return steps, actions.astype(np.float32), proprio


def qstats(values: np.ndarray):
    return {
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    out = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    equal = np.asarray(stats["min"]) == np.asarray(stats["max"])
    out[:, equal] = 0.0
    return out.astype(np.float32)


def main():
    start = time.time()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_manifest = {
        "dataset_repo": "openvla/modified_libero_rlds",
        "split_seed": SEED,
        "suites": list(SUITES),
        "episode_split": "per task, seeded selection of ceil(10%) heldout episodes",
        "caps": {"train_per_task": TRAIN_PER_TASK, "heldout_per_task": HELDOUT_PER_TASK},
        "suites_manifest": {},
    }
    for suite_idx, suite in enumerate(SUITES):
        print(f"[{suite}] metadata/statistics pass", flush=True)
        metas, task_to_indices = [], defaultdict(list)
        actions_all, proprio_all = [], []
        for idx, ep in enumerate(tfds.as_numpy(raw_episode_dataset(suite, skip_images=True))):
            steps, actions, proprio = transformed_arrays(ep)
            lang = task_from_language(steps[0]["language_instruction"])
            rec = {
                "episode_index": idx,
                "episode_id": episode_id(ep["episode_metadata"], idx),
                "task": lang,
                "transitions": int(len(actions)),
            }
            metas.append(rec)
            task_to_indices[lang].append(idx)
            actions_all.append(actions)
            proprio_all.append(proprio)
        actions_all, proprio_all = np.concatenate(actions_all), np.concatenate(proprio_all)
        stats = {
            "action": qstats(actions_all),
            "proprio": qstats(proprio_all),
            "num_transitions": int(len(actions_all)),
            "num_trajectories": len(metas),
        }
        rng = np.random.default_rng(SEED + suite_idx)
        split_by_episode = {}
        for task, indices in sorted(task_to_indices.items()):
            perm = np.asarray(indices)[rng.permutation(len(indices))]
            heldout_n = max(1, int(math.ceil(0.10 * len(indices))))
            heldout = set(int(x) for x in perm[-heldout_n:])
            for idx in indices:
                split_by_episode[idx] = "heldout" if idx in heldout else "train"
        for rec in metas:
            rec["split"] = split_by_episode[rec["episode_index"]]

        suite_out = OUT_ROOT / suite
        suite_out.mkdir(parents=True, exist_ok=True)
        (suite_out / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
        (suite_out / "episodes.json").write_text(json.dumps(metas, indent=2, sort_keys=True))

        buffers = {"train": defaultdict(list), "heldout": defaultdict(list)}
        counts = defaultdict(int)
        limits = {"train": TRAIN_PER_TASK, "heldout": HELDOUT_PER_TASK}
        print(f"[{suite}] materialization pass over {len(metas)} episodes", flush=True)
        for idx, ep in enumerate(tfds.as_numpy(raw_episode_dataset(suite, skip_images=False))):
            split = split_by_episode[idx]
            steps, actions, proprio = transformed_arrays(ep)
            actions_norm = normalize(actions, stats["action"])
            proprio_norm = normalize(proprio, stats["proprio"])
            lang = task_from_language(steps[0]["language_instruction"])
            take = min(len(actions) - 7, limits[split] - counts[(split, lang)])
            if take <= 0:
                continue
            for t in range(take):
                buffers[split]["primary"].append(steps[t]["observation"]["image"])
                buffers[split]["wrist"].append(steps[t]["observation"]["wrist_image"])
                buffers[split]["actions"].append(actions_norm[t : t + 8])
                buffers[split]["proprio"].append(proprio_norm[t])
                buffers[split]["language"].append(lang)
                buffers[split]["task"].append(lang)
                buffers[split]["episode_id"].append(episode_id(ep["episode_metadata"], idx))
                buffers[split]["timestep"].append(t)
                counts[(split, lang)] += 1

        split_manifest = {}
        for split, data in buffers.items():
            if not data["actions"]:
                raise RuntimeError(f"No samples materialized for {suite}/{split}")
            arrays = {
                "primary": np.stack(data["primary"]).astype(np.uint8),
                "wrist": np.stack(data["wrist"]).astype(np.uint8),
                "actions": np.stack(data["actions"]).astype(np.float32),
                "proprio": np.stack(data["proprio"]).astype(np.float32),
                "language": np.asarray(data["language"], dtype=object),
                "task": np.asarray(data["task"], dtype=object),
                "episode_id": np.asarray(data["episode_id"], dtype=object),
                "timestep": np.asarray(data["timestep"], dtype=np.int32),
            }
            for key, value in arrays.items():
                np.save(suite_out / f"{split}_{key}.npy", value, allow_pickle=key in {"language", "task", "episode_id"})
            split_manifest[split] = {
                "count": int(len(arrays["actions"])),
                "tasks": {task: int(n) for (sp, task), n in counts.items() if sp == split},
                "arrays": {key: str(suite_out / f"{split}_{key}.npy") for key in arrays},
            }
        all_manifest["suites_manifest"][suite] = {"statistics": stats, "splits": split_manifest}
        (RUN_ROOT / "01_dataset" / "split_manifest.json").write_text(json.dumps(all_manifest, indent=2, sort_keys=True))
        print(f"[{suite}] done", split_manifest, flush=True)
    all_manifest["elapsed_sec"] = time.time() - start
    (RUN_ROOT / "01_dataset" / "split_manifest.json").write_text(json.dumps(all_manifest, indent=2, sort_keys=True))
    print(json.dumps({"elapsed_sec": all_manifest["elapsed_sec"], "suites": list(SUITES)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
