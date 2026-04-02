import random
import glob
import re
from pathlib import Path


import numpy as np
from einops import rearrange
from PIL import Image
import torch
import torchvision.io

def seed_everything(seed: int | None = None) -> None:
    "Copy from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/platforms/interface.py#L170"
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_current_device() -> torch.device:
    return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")


def save_video(tensors: torch.Tensor | list[torch.Tensor], save_path: str, fps: int = 30) -> None:
    if not isinstance(tensors, list):
        tensors = [tensors]

    processed_tensors = []
    for tensor in tensors:
        if tensor.dtype != torch.uint8:
            raise ValueError("Input Tensor dtype must be uint8")
        if tensor.dim() != 4:
            raise ValueError(
                f"Input Tensor must be 4-dimensional (t, c, h, w), but got {tensor.dim()}")

        processed_tensors.append(tensor)

    final_tensor = torch.cat(processed_tensors, dim=3)

    if final_tensor.shape[0] == 1:
        image_tensor = rearrange(final_tensor, "1 c h w -> h w c")
        img = Image.fromarray(image_tensor.cpu().numpy())
        img.save(save_path)
    else:
        video_tensor = rearrange(final_tensor, "t c h w -> t h w c")
        torchvision.io.write_video(save_path, video_tensor, fps=fps)


def _dynamic_resize_from_bucket(image: Image, basesize: int = 512):
    from src.dataset.bucket_util import BucketGroup
    from src.config import ExpConfig, generate_video_image_bucket
    from typing import Tuple
    import math
    import torchvision.transforms.functional as TF

    def resize_center_crop(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """等比缩放到 >= 目标尺寸，再中心裁剪到目标尺寸。（PIL输入/输出）"""
        w, h = img.size  # PIL: (width, height)
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
        img = TF.resize(img, (resize_h, resize_w),
                        interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, target_size)
        return img

    bucket_config = generate_video_image_bucket(
        basesize=basesize, min_temporal=56, max_temporal=56, bs_img=4, bs_vid=4, bs_mimg=8, min_items=2, max_items=2
    )
    bucket_group = BucketGroup(bucket_config)
    img_w, img_h = image.size
    bucket = bucket_group.find_best_bucket((1, 1, img_h, img_w))
    target_height, target_width = bucket[-2], bucket[-1]  # (height, width)
    img_proc = resize_center_crop(image, (target_height, target_width))
    return img_proc
