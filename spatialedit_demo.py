# gr_edit_demo.py
import os
import math
from pathlib import Path
from typing import Optional, Tuple, List

import argparse
from PIL import Image
import gradio as gr
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import gc

from peft import PeftModel

# ==== src imports (与你的原始代码保持一致) ====
from src.models import load_dit, load_pipeline
from src.config import load_config_class_from_pyfile
from src.utils import _dynamic_resize_from_bucket, seed_everything


device = torch.device("cuda:1")

negative_prompt = ""

seed_everything(42)

# ---- 加载配置与模型 ----
config_path = "configs/spatialedit_base_config.py"
ckpt_path_PT = "your_base_path/SpatialEdit_CKPT/CKPT_PT"
ckpt_path_CT = "your_base_path/SpatialEdit_CKPT/CKPT_CT/lora"

config_class = load_config_class_from_pyfile(config_path)
cfg = config_class()

cfg.training_mode = False
cfg.use_fsdp_inference = False
cfg.hsdp_shard_dim = 1
cfg.dit_ckpt_type = "pt"
cfg.dit_ckpt = ckpt_path_PT
print(f"successfully load dit full param and merge it with {ckpt_path_PT}")
dit = load_dit(cfg, device=device)
dit.requires_grad_(False)
dit.eval()
dit = PeftModel.from_pretrained(dit, ckpt_path_CT)
dit = dit.merge_and_unload()
print(f"successfully load dit lora and merge it with {ckpt_path_CT}")

pipeline = load_pipeline(cfg, dit, device)

gen = torch.Generator(device=pipeline.transformer.device).manual_seed(int(42))

@torch.no_grad()
def run(
    image: Image.Image,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int,
    neg_prompt: str,
    basesize: int = 1024,
):
    if image is None and prompt.strip() == "":
        return None

    if image is None:
        prompts = [prompt]
        negative_prompt = [neg_prompt]
        images = None
    else:
        img_proc = _dynamic_resize_from_bucket(image, basesize=basesize)
        width, height = img_proc.size
        images = [img_proc]

        # 封装 prompt（与原始脚本对齐）
        image_tokens = "<image>\n"
        prompts = [f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"]
        negative_prompt = [
            f"<|im_start|>user\n{image_tokens}{neg_prompt}<|im_end|>\n"]

    # import ipdb; ipdb.set_trace()
    with torch.inference_mode():
        out_images = pipeline(
            prompt=prompts,
            negative_prompt=negative_prompt,
            images=images,
            height=height,
            width=width,
            num_frames=1,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            num_videos_per_prompt=1,
            output_type="pt",
            return_dict=False,
            enable_denormalization=cfg.enable_denormalization,
            # drop_vit_feature=True,
        )
    out_images = (out_images[0, -1, 0] * 255).to(torch.uint8).cpu()
    out_images = Image.fromarray(out_images.permute(1, 2, 0).numpy())
    return out_images


item_list = [
    ("i2i", "JD_Dog.jpeg", "Rotate the dog to show the front left side view.", 1024),
]

for idx, (task, image_path, prompt, basesize) in enumerate(item_list):

    if task == 't2i':
        height = basesize
        width = basesize
        image = None
    else:
        image = Image.open(os.path.join("validation", image_path))
        height = None
        width = None
    img = run(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        steps=30,
        guidance=5.0,
        seed=42,
        neg_prompt=negative_prompt,
        basesize=basesize,
    )
    img.save(f"output_image_edit_{device}_{idx}.png")
