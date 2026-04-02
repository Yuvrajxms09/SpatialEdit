# Adapted from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/distributed/communication_op.py

from typing import Any
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from src.distributed.parallel_states import get_parallel_state


class DistributedAutograd:
    """Collection of autograd functions for distributed operations.

    This class provides custom autograd functions for distributed operations like all_reduce,
    all_gather, and all_to_all. Each operation is implemented as a static inner class with
    proper forward and backward implementations.
    """

    class AllReduce(torch.autograd.Function):
        """Differentiable all_reduce operation.

        The gradient of all_reduce is another all_reduce operation since the operation
        combines values from all ranks equally.
        """

        @staticmethod
        def forward(ctx: Any,
                    group: ProcessGroup,
                    input_: Tensor,
                    op: dist.ReduceOp | None = None) -> Tensor:
            ctx.group = group
            ctx.op = op
            output = input_.clone()
            dist.all_reduce(output, group=group, op=op)
            return output

        @staticmethod
        def backward(ctx: Any,
                     grad_output: Tensor) -> tuple[None, Tensor, None]:
            grad_output = grad_output.clone()
            dist.all_reduce(grad_output, group=ctx.group, op=ctx.op)
            return None, grad_output, None

    class AllGather(torch.autograd.Function):
        """Differentiable all_gather operation.

        The operation gathers tensors from all ranks and concatenates them along a specified dimension.
        The backward pass uses reduce_scatter to efficiently distribute gradients back to source ranks.
        """

        @staticmethod
        def forward(ctx: Any, group: ProcessGroup, input_: Tensor,
                    world_size: int, dim: int) -> Tensor:
            ctx.group = group
            ctx.world_size = world_size
            ctx.dim = dim
            ctx.input_shape = input_.shape

            input_size = input_.size()
            output_size = (input_size[0] * world_size, ) + input_size[1:]
            output_tensor = torch.empty(output_size,
                                        dtype=input_.dtype,
                                        device=input_.device)

            dist.all_gather_into_tensor(output_tensor, input_, group=group)

            output_tensor = output_tensor.reshape((world_size, ) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
            return output_tensor

        @staticmethod
        def backward(ctx: Any,
                     grad_output: Tensor) -> tuple[None, Tensor, None, None]:
            # Split the gradient tensor along the gathered dimension
            dim_size = grad_output.size(ctx.dim) // ctx.world_size
            grad_chunks = grad_output.reshape(grad_output.shape[:ctx.dim] +
                                              (ctx.world_size, dim_size) +
                                              grad_output.shape[ctx.dim + 1:])
            grad_chunks = grad_chunks.movedim(ctx.dim, 0)

            # Each rank only needs its corresponding gradient
            grad_input = torch.empty(ctx.input_shape,
                                     dtype=grad_output.dtype,
                                     device=grad_output.device)
            dist.reduce_scatter_tensor(grad_input,
                                       grad_chunks.contiguous(),
                                       group=ctx.group)

            return None, grad_input, None, None

    class AllToAll4D(torch.autograd.Function):
        """Differentiable all_to_all operation specialized for 4D tensors.

        This operation is particularly useful for attention operations where we need to
        redistribute data across ranks for efficient parallel processing.

        The operation supports two modes:
        1. scatter_dim=2, gather_dim=1: Used for redistributing attention heads
        2. scatter_dim=1, gather_dim=2: Used for redistributing sequence dimensions
        """

        @staticmethod
        def forward(ctx: Any, group: ProcessGroup, input_: Tensor,
                    world_size: int, scatter_dim: int,
                    gather_dim: int) -> Tensor:
            ctx.group = group
            ctx.world_size = world_size
            ctx.scatter_dim = scatter_dim
            ctx.gather_dim = gather_dim

            if world_size == 1:
                return input_

            assert input_.dim(
            ) == 4, f"input must be 4D tensor, got {input_.dim()} and shape {input_.shape}"

            if scatter_dim == 2 and gather_dim == 1:
                bs, shard_seqlen, hc, hs = input_.shape
                seqlen = shard_seqlen * world_size
                shard_hc = hc // world_size

                input_t = input_.reshape(bs, shard_seqlen, world_size, shard_hc,
                                         hs).transpose(0, 2).contiguous()
                output = torch.empty_like(input_t)

                dist.all_to_all_single(output, input_t, group=group)

                output = output.reshape(seqlen, bs, shard_hc,
                                        hs).transpose(0, 1).contiguous()
                output = output.reshape(bs, seqlen, shard_hc, hs)

                return output
            elif scatter_dim == 1 and gather_dim == 2:
                bs, seqlen, shard_hc, hs = input_.shape
                hc = shard_hc * world_size
                shard_seqlen = seqlen // world_size

                input_t = input_.reshape(bs, world_size, shard_seqlen, shard_hc,
                                         hs)
                input_t = input_t.transpose(0, 3).transpose(0, 1).contiguous()
                input_t = input_t.reshape(world_size, shard_hc, shard_seqlen,
                                          bs, hs)

                output = torch.empty_like(input_t)
                dist.all_to_all_single(output, input_t, group=group)

                output = output.reshape(hc, shard_seqlen, bs, hs)
                output = output.transpose(0, 2).contiguous()
                output = output.reshape(bs, shard_seqlen, hc, hs)

                return output
            else:
                raise RuntimeError(
                    f"Invalid scatter_dim={scatter_dim}, gather_dim={gather_dim}. "
                    f"Only (scatter_dim=2, gather_dim=1) and (scatter_dim=1, gather_dim=2) are supported."
                )

        @staticmethod
        def backward(
                ctx: Any,
                grad_output: Tensor) -> tuple[None, Tensor, None, None, None]:
            if ctx.world_size == 1:
                return None, grad_output, None, None, None

            # For backward pass, we swap scatter_dim and gather_dim
            output = DistributedAutograd.AllToAll4D.apply(
                ctx.group, grad_output, ctx.world_size, ctx.gather_dim,
                ctx.scatter_dim)
            return None, output, None, None, None


def sequence_parallel_all_to_all_4D(input_: torch.Tensor,
                                    scatter_dim: int = 2,
                                    gather_dim: int = 1) -> torch.Tensor:
    """All-to-all communication of 4D tensors (e.g. QKV matrices) across sequence parallel group."""
    return DistributedAutograd.AllToAll4D.apply(
        get_parallel_state().sp_group,
        input_,
        get_parallel_state().sp_size,
        scatter_dim,
        gather_dim,
    )


def sequence_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return DistributedAutograd.AllGather.apply(
        get_parallel_state().sp_group,
        input_,
        get_parallel_state().sp_size,
        dim,
    )


def broadcast_within_sp_group(input_: torch.Tensor):
    src = get_parallel_state().sp_group_id * get_parallel_state().sp_size
    dist.broadcast(input_, src=src, group=get_parallel_state().sp_group)


def broadcast_item(item, src: int = 0):
    if not dist.is_initialized():
        return item

    item_list = [item]
    dist.broadcast_object_list(item_list, src=src)
    return item_list[0]
