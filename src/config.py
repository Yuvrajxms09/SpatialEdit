from dataclasses import dataclass, field
import json
import importlib.util
import inspect
from pathlib import Path
from typing import Type

from src.distributed.parallel_states import print_rank0


@dataclass
class ExpConfig:
    seed: int = 42
    exp_name: str = "test"
    output_dir: str = "./output"

    # Resume
    resume_from_checkpoint: str = None
    resume_optimizer: bool = True
    resume_dataloader: bool = True
    auto_resume: bool = False
    resume_lora_path: str = None

    # DIT
    dit_ckpt: str = None
    dit_ckpt_type: str = "pt"  # "safetensor" or "pt"
    dit_arch_config: str = None
    dit_precision: str = "bf16"
    is_repa: bool = False
    repa_layer: int = 20
    repa_lambda: float = 0.5
    repa_aligh: str = 'patch'

    # VAE
    vae_ckpt: str = None
    vae_precision: str = "fp16"
    enable_denormalization: bool = True

    # Text Encoder
    text_encoder_arch_config: str = None
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 512

    # Data
    train_image_data_files: str = None
    train_video_data_files: str = None
    train_image_caption_keys: list[str] = None
    train_image_caption_sampling_prob: list[float] = None
    train_video_caption_keys: list[str] = None
    train_video_caption_sampling_prob: list[float] = None
    video_sampling_prob: float = 1
    bucket_configs: list[tuple[int, int, int, int]] = None
    bucket_configs_options: list[tuple[int, int, int, int]] = None
    bucket_configs_options_prob: float = 0.5,
    prioritize_frame_matching: bool = True
    ensure_divisible_shards: bool = True
    shuffle: bool = True
    num_workers: int = 2
    fps: int = -1
    rec_aug_rate: float = 0.0
    score_rate: float = 0.0
    use_new_caption: bool = False
    train_multiple_images_data_files_space_edit: list[str] = None
    train_multiple_images_data_files_space_edit_prob: float = 0.0
    train_multiple_images_data_files_space_edit_sub_prob: list[float] = None

    # Training
    weighting_scheme: str = "lognorm"
    train_flow_shift: int = 1
    train_flow_base: int = 1
    cfg_rate: float = 0.1
    use_lora: bool = False
    lora_rank: int = 16
    use_ema: bool = False
    ema_decay: float = 0.999

    micro_batch_size: int = 1
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    data_check_interval: int = 100
    validation_steps: list[int] = field(default_factory=lambda: [1, 10, 100])
    validation_interval: int = 100
    checkpoint_interval: int = 1000
    grad_check_interval: int = 1000
    gc_interval: int = 1000
    log_interval: int = 10

    optimizer_name: str = 'adamw'
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0
    adam_epsilon: float = 1e-10
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100

    enable_activation_checkpointing: bool = False
    activation_checkpointing_type: str = "full"  # "full" or "block_skip"
    # for "block_skip", checkpoint every N-th
    activation_checkpointing_skip_interval: int = 2

    drop_vit_feature: bool = False
    use_kl: bool = False
    kl_coef: float = 1.0
    drop_ab_test: bool = True

    # Validation
    val_data_files: str = None
    val_noise_scheduler: str = "flow_euler_discrete"
    val_flow_shift: int = 1
    val_batch_size: int = 1
    val_width: int = 256
    val_height: int = 256
    val_basesize: int = 1024
    val_num_frames: int = 33
    val_num_inference_steps: int = 30
    val_guidance_scale: float = 7.5
    val_max_samples: int = None

    # Parallelism
    sp_size: int = 1

    # FSDP2
    training_mode: bool = True
    hsdp_shard_dim: int = 1
    reshard_after_forward: bool = False  # zero2=False, zero3=True
    use_fsdp_inference: bool = False
    cpu_offload: bool = False
    pin_cpu_memory: bool = False
    enable_torch_compile: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.resume_from_checkpoint and self.dit_ckpt:
            raise ValueError(
                "Cannot specify both 'resume_from_checkpoint' and 'dit_ckpt'. Choose one.")

    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def load_config_class_from_pyfile(file_path: str) -> Type[ExpConfig]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        # The condition ensures we find a subclass, not ExpConfig itself.
        if issubclass(obj, ExpConfig) and obj is not ExpConfig:
            print_rank0(
                f"Dynamically loaded config class: '{obj.__name__}' from '{file_path}'")
            return obj

    raise ValueError(
        f"No class inheriting from 'ExpConfig' was found in '{file_path}'.")


def _generate_hw_buckets(base_height=256, base_width=256, step_width=16, step_height=16, max_ratio=4.0) -> list[tuple[int, int, int, int, int]]:
    """Generate dimension buckets based on aspect ratios"""
    buckets = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            ratio = height / width
            buckets.append((1, 1, 1, height, width))
        # Try to increase width or decrease height
        if height * (width + step_width) <= target_pixels:
            width += step_width
        else:
            height -= step_height

    return buckets


def generate_video_image_bucket(basesize=256, min_temporal=65, max_temporal=129, bs_img=8, bs_vid=1, bs_mimg=4, min_items=1, max_items=1):
    # (batch_size, num_items, num_frames, height, width)
    assert basesize in [
        256, 512, 768, 1024], f"[generate_video_image_bucket] wrong basesize {basesize}"
    bucket_list = []
    # base_bucket_list = [
    #     (1, 1, 1, 512, 128),  # 4:1
    #     (1, 1, 1, 128, 512),
    #     (1, 1, 1, 192, 352),  # 16:9
    #     (1, 1, 1, 352, 192),
    #     (1, 1, 1, 288, 224),  # 4:3
    #     (1, 1, 1, 224, 288),
    #     (1, 1, 1, 320, 208),  # 3:2
    #     (1, 1, 1, 208, 320),
    #     (1, 1, 1, 368, 176),  # 2:1
    #     (1, 1, 1, 176, 368),
    #     (1, 1, 1, 256, 256),  # 1:1
    # ]

    base_bucket_list = _generate_hw_buckets()
    # image
    for _bucket in base_bucket_list:
        bucket = list(_bucket)
        bucket[0] = bs_img
        bucket_list.append(bucket)
    # video
    for temporal in range(min_temporal, max_temporal+1, 8):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bs = (max_temporal + 1) // temporal * bs_vid
            bucket[0] = bs
            bucket[2] = temporal
            bucket_list.append(bucket)
    # multiple images
    for num_items in range(min_items, max_items+1):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bucket[0] = bs_mimg
            bucket[1] = num_items
            bucket_list.append(bucket)
    # spatial resize
    if basesize > 256:
        ratio = basesize // 256

        def resize(bucket, r):
            bucket[-2] *= r
            bucket[-1] *= r
            return bucket
        bucket_list = [resize(bucket, ratio) for bucket in bucket_list]
    return bucket_list
