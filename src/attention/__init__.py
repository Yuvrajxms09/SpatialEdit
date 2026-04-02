# Adapted from https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/attention

import os
import torch
from einops import rearrange

try:
    # Check for Flash Attention 3 installation path
    flash_attn3_path = os.getenv("FLASH_ATTN3_PATH")
    if flash_attn3_path:
        print(f"Using Flash Attention 3 from: {flash_attn3_path}")
        import sys
        sys.path.insert(0, flash_attn3_path)
        from flash_attn_interface import flash_attn_varlen_func
    else:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


from src.distributed.parallel_states import get_parallel_state
from src.distributed.communications import sequence_parallel_all_gather, sequence_parallel_all_to_all_4D


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1],
                             dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "flash_attn",
    *,
    causal: bool = False,
    softmax_scale: float = None,
    attn_kwargs: dict = None,
):
    """
    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads
    """
    assert backend in [
        "torch_spda", "flash_attn"], f"Unsupported attention backend: {backend}"
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Input tensors must be 4D"
    batch_size = q.shape[0]
    if backend == "torch_spda":
        # transpose to [bs, heads, seq_len, head_dim]
        q = rearrange(q, "b l h c -> b h l c")
        k = rearrange(k, "b l h c -> b h l c")
        v = rearrange(v, "b l h c -> b h l c")
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=softmax_scale)
        output = rearrange(output, "b h l c -> b l h c")
    elif backend == "flash_attn":
        cu_seqlens_q = attn_kwargs['cu_seqlens_q']
        cu_seqlens_kv = attn_kwargs['cu_seqlens_kv']
        max_seqlen_q = attn_kwargs['max_seqlen_q']
        max_seqlen_kv = attn_kwargs['max_seqlen_kv']
        x = flash_attn_varlen_func(
            q.view(q.shape[0] * q.shape[1], *q.shape[2:]),
            k.view(k.shape[0] * k.shape[1], *k.shape[2:]),
            v.view(v.shape[0] * v.shape[1], *v.shape[2:]),
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        output = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]

    return output

# TODO: support attention_mask


def distributed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "flash_attn",
    attn_kwargs: dict = None,
    ###
    replicated_q: torch.Tensor | None = None,
    replicated_k: torch.Tensor | None = None,
    replicated_v: torch.Tensor | None = None,
):
    """
    TODO: support attention mask
    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim()
    batch_size, seq_len, num_heads, head_dim = q.shape

    local_rank = get_parallel_state().rank_within_sp_group
    world_size = get_parallel_state().sp_size

    # Stack QKV
    # [3, batch_size, seq_len, num_heads, head_dim]
    qkv = torch.cat([q, k, v], dim=0)

    # Redistribute heads across sequence dimension
    qkv = sequence_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)

    # Concatenate with replicated QKV if provided
    if replicated_q is not None:
        assert replicated_k is not None and replicated_v is not None
        # [3, batch_size, seq_len, num_heads, head_dim]
        replicated_qkv = torch.cat(
            [replicated_q, replicated_k, replicated_v], dim=0)
        heads_per_rank = num_heads // world_size
        replicated_qkv = replicated_qkv[:, :, local_rank *
                                        heads_per_rank:(local_rank + 1) * heads_per_rank]
        qkv = torch.cat([qkv, replicated_qkv], dim=1)

    q, k, v = qkv.chunk(3, dim=0)

    output = attention(q, k, v, backend=backend, attn_kwargs=attn_kwargs)

    # Redistribute back if using sequence parallelism
    replicated_output = None
    if replicated_q is not None:
        replicated_output = output[:, seq_len * world_size:]
        output = output[:, :seq_len * world_size]
        # TODO: make this asynchronous
        replicated_output = sequence_parallel_all_gather(
            replicated_output.contiguous(), dim=2)

    output = sequence_parallel_all_to_all_4D(
        output, scatter_dim=1, gather_dim=2)
    return output, replicated_output


__all__ = [
    "attention",
    "distributed_attention",
]
