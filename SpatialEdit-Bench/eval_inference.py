import os
import time
import math
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import random
from typing import Any
from PIL import Image
from einops import rearrange
from peft import PeftModel

import torch
import torch.nn.functional as F
import torch.distributed as dist
from pathvalidate import sanitize_filename
import torchvision.transforms.functional as TF
from datasets import load_dataset
from tqdm import tqdm

from src.models import (
    load_dit, load_pipeline
)
from src.distributed.parallel_states import (
    init_distributed_environment_and_sequence_parallel,
    get_parallel_state, sp_enabled, clean_dist_env
)
from src.config import load_config_class_from_pyfile, ExpConfig
from src.utils import seed_everything, save_video
from src.dataset.bucket_util import BucketGroup
from src.utils import _dynamic_resize_from_bucket
from src.utils.logging import get_logger


def setup_distributed_training(args: Any) -> tuple[int, int, int, torch.device]:
    """Setup distributed training environment."""
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    init_distributed_environment_and_sequence_parallel(sp_size=args.sp_size)

    return local_rank, global_rank, world_size, device


def resize_crop(img: Image, target_size: tuple[int, int]):
    w, h = img.size  # 注意 PIL 是 (width, height)
    bh, bw = target_size
    if w == bw and h == bh:
        return img

    scale = max(bh / h, bw / w)
    resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

    # resize 和 crop
    img = img.resize((resize_w, resize_h), Image.LANCZOS)
    img = TF.center_crop(img, target_size)
    return img


def main(cfg: ExpConfig, args: Any):
    """Main training function."""
    # Setup distributed training
    local_rank, global_rank, world_size, device = setup_distributed_training(
        args)
    print(
        f"local_rank: {local_rank}, global_rank: {global_rank}, world_size: {world_size}, device: {device}")

    # Setup seed
    assert args.seed is not None, "Seed must be specified in the configuration."
    seed_everything(args.seed)

    # Setup directories
    assert args.save_path is not None, "Output directory must be specified in the configuration."

    if global_rank <= 0:
        os.makedirs(args.save_path, exist_ok=True)

    # Broadcast directories to all processes
    dist.barrier()
    print(f"barrier done")

    # Setup model and close fsdp
    cfg.use_lora = False
    cfg.training_mode = False
    cfg.use_fsdp_inference = False
    cfg.hsdp_shard_dim = 1
    if args.ckpt_path.endswith(".pth"):
        cfg.dit_ckpt_type = "pt"
        cfg.dit_ckpt = args.ckpt_path
        print(f"successfully load dit full param and merge it with {args.ckpt_path}")
    dit = load_dit(cfg, device=device)
    dit.requires_grad_(False)
    dit.eval()

    if not args.ckpt_path.endswith(".pth"):
        lora_path = args.ckpt_path
        dit = PeftModel.from_pretrained(dit, lora_path)
        dit = dit.merge_and_unload()
        print(f"successfully load dit lora and merge it with {lora_path}")

    # Setup validation pipeline
    pipeline = load_pipeline(cfg, dit, device)
    pipeline._progress_bar_config = {"disable": True}

    # Load dataset
    dataset_path = args.meta_file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Begin evaluation
    indices = [_ for _ in range(len(dataset))]
    indices = indices[global_rank::world_size]

    for idx in tqdm(indices, total=len(indices), desc=f"rank {global_rank}: "):
        # Setup noise generator
        noise_generator = torch.Generator(
            device=pipeline.transformer.device).manual_seed(args.seed)

        item = dataset[idx]
        image_path = os.path.join(args.bench_data_dir, item["image_path"])
        edit_task = item["type"]
        save_image_dir_id = item['image_id']
        image_key = item['edit_id']
        if edit_task == "camera":
            instruction_list = {
                # "en": item["prompt_human"],
                "en": item["prompt"],
            }
        else:
            instruction_list = {
                "en": item["prompt"],
            }

        for instruction_language, instruction in instruction_list.items():

            dir_path = args.save_path + \
                f"/fullset/{edit_task}/{instruction_language}/{save_image_dir_id}"
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(os.path.join(dir_path, f"{image_key}.png")):
                print(f"Rank {global_rank} - {image_key} exists, skip.")
                continue

            prompts = [
                f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n"]
            negative_prompt = [
                f"<|im_start|>user\n<image>\n{args.neg_prompt}<|im_end|>\n"]
            images = [Image.open(image_path).convert("RGB")]

            frame = args.frame
            height = args.height
            width = args.width

            images = [_dynamic_resize_from_bucket(
                img, basesize=args.basesize) for img in images]
            width, height = images[0].size

            with torch.inference_mode():
                videos = pipeline(
                    prompt=prompts,
                    negative_prompt=negative_prompt,
                    images=images,
                    height=height,
                    width=width,
                    num_frames=frame,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=noise_generator,
                    num_videos_per_prompt=1,
                    output_type="pt",
                    return_dict=False,
                    enable_denormalization=cfg.enable_denormalization,
                )

            suffix = ".png"
            for _, (video, prompt) in enumerate(zip(videos, [instruction])):
                video = (video * 255).to(torch.uint8).cpu()
                for vid in range(video.shape[0]):
                    if vid == video.shape[0] - 1:
                        image_tensor = rearrange(video[vid], "1 c h w -> h w c")
                        img = Image.fromarray(image_tensor.numpy())
                        img.save(os.path.join(args.save_path,
                                 dir_path, f"{image_key}{suffix}"))
                    else:
                        images[vid].save(os.path.join(
                            args.save_path, dir_path, f"{image_key}_src{suffix}"))
                with open(os.path.join(dir_path, f"{image_key}_prompt.txt"), "w") as f:
                    f.write(prompt)
            if edit_task == "camera":
                json_file_path = f"{image_path[:-4]}.json"
                filename = os.path.basename(json_file_path)
                dst_dir = os.path.join(args.save_path, dir_path, filename)
                import shutil
                shutil.copy2(json_file_path, dst_dir)
                gt_image_path = image_path.replace(".jpg", "_gt.jpg")
                shutil.copy2(gt_image_path, os.path.join(args.save_path, dir_path, f"{image_key}_gt{suffix}"))

    dist.barrier()
    print(f"Images Generated Done.")
    clean_dist_env()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="spatialedit")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to the ckpt file.")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Path to the save path.")
    parser.add_argument("--meta-file", type=str, required=True,
                        help="Path to the meta_file path.")
    parser.add_argument("--bench-data-dir", type=str, required=True,
                        help="Path to the meta_file path.")
    parser.add_argument("--basesize", type=int, default=512,
                        help="base size of width/height.")
    parser.add_argument("--width", type=int, default=512, help="width.")
    parser.add_argument("--height", type=int, default=512, help="height.")
    parser.add_argument("--frame", type=int, default=1, help="frame.")
    parser.add_argument("--num-inference-steps", type=int,
                        default=30, help="num-inference-steps.")
    parser.add_argument("--guidance-scale", type=float,
                        default=5.0, help="guidance-scale.")
    parser.add_argument("--neg-prompt", type=str,
                        default="", help="neg_prompt.")
    parser.add_argument("--seed", type=int, default=42, help="seed.")
    parser.add_argument("--sp-size", type=int, default=1, help="sp_size.")
    args = parser.parse_args()

    config_class = load_config_class_from_pyfile(args.config)
    cfg = config_class()
    main(cfg, args)
