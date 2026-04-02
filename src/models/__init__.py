import os
import glob
import torch
import torch.distributed as dist

from src.utils.logging import get_logger
from src.utils.constants import PRECISION_TO_TYPE
from src.utils.utils import build_from_config
from src.models.common.diffusion.pipelines import Pipeline
from typing import Generator
from tqdm import tqdm

_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501

def load_pipeline(cfg, dit, device: torch.device):
    # vae
    factory_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.vae_precision], "device": device}
    vae = build_from_config(cfg.vae_arch_config, **factory_kwargs)
    if getattr(cfg.vae_arch_config, "enable_feature_caching", False):
        vae.enable_feature_caching()

    # text_encoder
    factory_kwargs = {
        'torch_dtype': PRECISION_TO_TYPE[cfg.text_encoder_precision], "device": device}
    tokenizer, text_encoder = build_from_config(
        cfg.text_encoder_arch_config, **factory_kwargs)

    # scheduler
    scheduler = build_from_config(cfg.scheduler_arch_config)

    pipeline = Pipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=dit,
        scheduler=scheduler,
        args=cfg,
    )

    pipeline = pipeline.to(device)
    return pipeline

def pt_weights_iterator(hf_weights_files: list[str]) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    device = "cpu"
    enable_tqdm = not torch.distributed.is_initialized(
    ) or torch.distributed.get_rank() == 0
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(bin_file, map_location=device, weights_only=True)
        yield from state.items()
        del state

def load_dit(cfg, device: torch.device) -> torch.nn.Module:
    """Load DiT model with FSDP support."""
    logger = get_logger()

    state_dict = None
    if cfg.dit_ckpt is not None:
        logger.info(
            f"Loading model from: {cfg.dit_ckpt}, type: {cfg.dit_ckpt_type}")

        if cfg.dit_ckpt_type == "pt":
            pt_files = [cfg.dit_ckpt]
            state_dict = dict(pt_weights_iterator(pt_files))
            if "model" in state_dict:
                state_dict = state_dict["model"]
        else:
            raise ValueError(
                f"Unknown dit_ckpt_type: {cfg.dit_ckpt_type}, must be 'safetensor' or 'pt'")

    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    model_kwargs = {'dtype': dtype, 'device': device, 'args': cfg}
    model = build_from_config(cfg.dit_arch_config, **model_kwargs)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        # Debug mode
        model.to(device=device)

    if state_dict is not None:
        # filter unused params
        load_state_dict = {}
        for k, v in state_dict.items():
            if not cfg.is_repa and 'repa' in k:
                continue

            if k == "img_in.weight" and model.img_in.weight.shape != v.shape:
                logger.info(
                    f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                v_new = v.new_zeros(model.img_in.weight.shape)
                v_new[:, :v.shape[1], :, :, :] = v
                v = v_new

            load_state_dict[k] = v
        model.load_state_dict(load_state_dict, strict=True)

    if cfg.use_lora:
        model = model.to(dtype).to(device)
        return model.eval()

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

    # Ensure consistent dtype
    param_dtypes = {param.dtype for param in model.parameters()}
    if len(param_dtypes) > 1:
        logger.warning(
            f"Model has mixed dtypes: {param_dtypes}. Converting to {dtype}")
        model = model.to(dtype)

    return model.eval()


__all__ = [
    "load_vae",
    "load_pipeline",
]
