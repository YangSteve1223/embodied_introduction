import pickle
import time
import os
from types import SimpleNamespace

import numpy as np
import torch

import experiments.robot.openvla_utils as openvla_utils
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


CHECKPOINT = os.environ.get(
    "OPENVLA_CHECKPOINT",
    "moojink/openvla-7b-oft-finetuned-libero-spatial",
)
SAMPLE_PATH = (
    "/share/yangpengju-local/openvla-oft-repro/src/openvla-oft/"
    "experiments/robot/libero/sample_libero_spatial_observation.pkl"
)


def main() -> None:
    # For the remote Hub ID, avoid an extra HfApi model_info request, which can
    # hang on restricted server egress. Local checkpoints use the official
    # utility path and load all auxiliary files from disk.
    if CHECKPOINT.startswith("moojink/"):
        openvla_utils.model_is_on_hf_hub = lambda _: True

    cfg = SimpleNamespace(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        unnorm_key="libero_spatial_no_noops",
        lora_rank=32,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
    )

    print("=== phase B sample inference ===", flush=True)
    print("checkpoint:", CHECKPOINT, flush=True)
    print("cuda_available:", torch.cuda.is_available(), flush=True)
    print("visible_gpu_count:", torch.cuda.device_count(), flush=True)

    load_start = time.perf_counter()
    print("before get_vla", flush=True)
    vla = openvla_utils.get_vla(cfg)
    print("after get_vla", flush=True)
    processor = openvla_utils.get_processor(cfg)
    action_head = openvla_utils.get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = openvla_utils.get_proprio_projector(
        cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("vla_type:", type(vla).__name__, flush=True)
    print("processor_type:", type(processor).__name__, flush=True)
    print("action_head_type:", type(action_head).__name__, flush=True)
    print("proprio_projector_type:", type(proprio_projector).__name__, flush=True)
    print("load_time_sec:", time.perf_counter() - load_start, flush=True)

    with open(SAMPLE_PATH, "rb") as f:
        observation = pickle.load(f)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    infer_start = time.perf_counter()
    actions = openvla_utils.get_vla_action(
        cfg,
        vla,
        processor,
        observation,
        observation["task_description"],
        action_head,
        proprio_projector,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    actions_np = np.asarray([
        x.detach().float().cpu().numpy() if torch.is_tensor(x) else x
        for x in actions
    ])

    print("action_chunk_shape:", actions_np.shape, flush=True)
    print("action_chunk_dtype:", actions_np.dtype, flush=True)
    print("inference_time_sec:", time.perf_counter() - infer_start, flush=True)
    print(
        "inference_peak_memory_gb:",
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available() else None,
        flush=True,
    )
    print("finite:", bool(np.isfinite(actions_np).all()), flush=True)
    print("min:", float(actions_np.min()), flush=True)
    print("max:", float(actions_np.max()), flush=True)
    print("mean:", float(actions_np.mean()), flush=True)
    print("std:", float(actions_np.std()), flush=True)

    assert actions_np.shape == (NUM_ACTIONS_CHUNK, 7)
    assert np.isfinite(actions_np).all()
    print("OPENVLA_OFT_SAMPLE_INFERENCE_OK", flush=True)


if __name__ == "__main__":
    main()
