#!/usr/bin/env python3
"""Download the official modified_libero_rlds subsets to local server storage."""
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(os.environ.get("REPRO_ROOT", "/share/yangpengju-local/openvla-oft-repro"))
DATA_ROOT = ROOT / "datasets" / "modified_libero_rlds"
HF_HOME = ROOT / "cache" / "huggingface"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

repo_id = "openvla/modified_libero_rlds"
patterns = [
    "README.md",
    "libero_spatial_no_noops/**",
    "libero_object_no_noops/**",
    "libero_goal_no_noops/**",
    "libero_10_no_noops/**",
]

start = time.time()
print(json.dumps({"repo_id": repo_id, "patterns": patterns, "data_root": str(DATA_ROOT)}, sort_keys=True), flush=True)
api = HfApi()
info = api.dataset_info(repo_id)
print(json.dumps({"repo_sha": info.sha, "siblings": len(info.siblings)}, sort_keys=True), flush=True)

local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(DATA_ROOT),
    local_dir_use_symlinks=False,
    allow_patterns=patterns,
    resume_download=True,
)

files = []
for path in DATA_ROOT.rglob("*"):
    if path.is_file() and ".cache" not in path.parts:
        files.append({"path": str(path), "bytes": path.stat().st_size})
manifest = {
    "repo_id": repo_id,
    "repo_sha": info.sha,
    "allow_patterns": patterns,
    "local_dir": local_dir,
    "file_count": len(files),
    "bytes": sum(item["bytes"] for item in files),
    "elapsed_sec": time.time() - start,
    "files": files,
}
(ROOT / "runs" / "post_training_20260922" / "01_dataset" / "download_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True)
)
print(json.dumps({k: manifest[k] for k in ("repo_sha", "file_count", "bytes", "elapsed_sec")}, sort_keys=True), flush=True)
