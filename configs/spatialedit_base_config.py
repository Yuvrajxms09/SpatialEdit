import os
from dataclasses import dataclass, field
from src.config import ExpConfig, generate_video_image_bucket


@dataclass
class PretrainT2I256P(ExpConfig):
    seed: int = 42

    # DIT
    # dit_ckpt: str = "your_base_path/model"
    dit_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.dit.Transformer3DModel",
        "params": {
            "hidden_size": 4096,
            "in_channels": 16,
            "heads_num": 32,
            "mm_double_blocks_depth": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "rope_dim_list": [16, 56, 56],
            "text_states_dim": 4096,
            "rope_type": "rope",
            "dit_modulation_type": "wanx",
            "unpatchify_new": True,
        }
    }
    )
    dit_precision: str = "bf16"
    is_repa: bool = False
    repa_layer: int = 13
    repa_lambda: float = 0.5
    repa_aligh: str = "patch"

    # VAE
    vae_arch_config: dict = field(default_factory=lambda: {
        "target": "src.models.mmdit.vae.WanxVAE",
        "pretrained": "your_base_path/model/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    }
    )
    vae_precision: str = "bf16"
    enable_denormalization: bool = False

    # Text Encoder
    text_encoder_arch_config: dict = field(
        default_factory=lambda: {
            "target": "src.models.mmdit.text_encoder.load_text_encoder",
            "params": {
                "text_encoder_ckpt": "your_base_path/model/Qwen3-VL-8B-Instruct",
            },
        }
    )
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048

    # scheduler
    scheduler_arch_config: dict = field(
        default_factory=lambda: {
            "target": "src.models.common.diffusion.schedulers.FlowMatchDiscreteScheduler",
            "params": {
                "num_train_timesteps": 1000,
                "shift": 1.5,
            },
        }
    )

    use_lora: bool = True
    lora_rank: int = 16
    # Parallism
    sp_size: int = 1

    # FSDP
    reshard_after_forward: bool = False
    # hsdp_shard_dim: int = 64
    hsdp_shard_dim: int = 1
